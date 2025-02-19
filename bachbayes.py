import csv
import copy
import numpy as np
import glob
import io
import mido
import os
import re
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import zipfile
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urljoin, urlparse


# Neural network models
class RNN(nn.Module):

	def __init__(self, n_dim, n_layers, n_hidden, unit_type, dev, batch_size=None):
		
		super(RNN, self).__init__()

		self.n_dim     = n_dim
		self.n_layers  = n_layers
		self.n_hidden  = n_hidden
		self.unit_type = unit_type
		self.dev       = dev

		if self.unit_type.lower() == 'elman':
			self.rnn = nn.RNN(n_dim, n_hidden, n_layers, batch_first=True).to(self.dev)
		elif self.unit_type.lower() == 'gru':
			self.rnn = nn.GRU(n_dim, n_hidden, n_layers, batch_first=True).to(self.dev)
		elif self.unit_type.lower() == 'lstm':
			self.rnn = nn.LSTM(n_dim, n_hidden, n_layers, batch_first=True).to(self.dev)
		else:
			print(f'Unit type {self.unit_type} not recognised')

		self.out_obs  = nn.Linear(n_hidden, n_dim, bias=True).to(self.dev)
		self.sig_obs  = nn.Sigmoid().to(self.dev)
		self.out_pred = nn.Linear(n_hidden, n_dim, bias=True).to(self.dev)
		self.sig_pred = nn.Sigmoid().to(self.dev)

		if batch_size is not None:
			self.init_hidden(batch_size)
		else:
			self.hidden = None

	def init_hidden(self, batch_size):
		h_size = [self.n_layers, batch_size, self.n_hidden]
		self.hidden = torch.zeros(*h_size).to(self.dev)

	def forward(self, x):

		self.init_hidden(x.shape[0])
		x, self.hidden = self.rnn(x, self.hidden)
		x_obs  = self.sig_obs(self.out_obs(x))
		x_pred = self.sig_pred(self.out_pred(x))

		return x_obs, x_pred


class FeedForwardNN(nn.Module):

	def __init__(self, n_dim, n_hidden_layers, n_hidden_units, dev):
		
		super(FeedForwardNN, self).__init__()

		self.n_dim           = n_dim
		self.n_hidden_layers = n_hidden_layers
		self.n_hidden_units  = n_hidden_units
		self.dev             = dev

		layers = [nn.Linear(self.n_dim, self.n_hidden_units), nn.ReLU()]
		for _ in range(self.n_hidden_layers):
			layers.append(nn.Linear(self.n_hidden_units, self.n_hidden_units))
			layers.append(nn.ReLU())
		layers += [nn.Linear(self.n_hidden_units, self.n_dim), nn.Sigmoid()]

		self.feedforward = nn.Sequential(*layers).to(self.dev)

	def forward(self, x):
		x = self.feedforward(x) 
		# x serves as either inferred value or prediction:
		return x, x 


# Model object
class Bachmodel():

	def __init__(self, pars, dev=None):
		super(Bachmodel, self).__init__()

		if dev is None:
			dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.pars = pars
		self.dev  = dev
		self.chromatic       = pars['chromatic']
		self.noise           = pars['noise']
		self.train_path      = os.path.join(pars['datapath'], 'training')
		self.validation_path = os.path.join(pars['datapath'], 'validation')
		self.test_path       = os.path.join(pars['datapath'], 'test')

		n_dim = 12 if self.chromatic else 108
		n_layers = pars['n_layers']
		n_hidden = pars['n_hidden']
		mod_type = pars['mod_type']

		if mod_type.lower() == 'feedforward':
			self.net = FeedForwardNN(n_dim, n_layers, n_hidden, dev=dev)
		else:
			self.net = RNN(n_dim, n_layers, n_hidden, mod_type, dev=dev)

		modname =  f'{mod_type}_nhidden-{n_hidden:d}_nlayers-{n_layers:d}'
		modname += f'{"_chromatic" if self.chromatic else ""}'
		modname += f'_noise{self.noise}'.replace('.', 'p')
		self.modname = modname
		self.modpath = f'./models/{modname}.pth'

	def load_weights(self, suffix=''):
		weightpath = self._compose_weightpath_(suffix)
		self.net.load_state_dict(torch.load(weightpath))

	def save_weights(self, suffix=''):
		weightpath = self._compose_weightpath_(suffix)
		torch.save(self.net.state_dict(), weightpath)

	def trained_weights_exists(self, suffix=''):
		weightpath = self._compose_weightpath_(suffix)
		return os.path.exists(weightpath)

	def train(self, lr, chunk_size, batch_size, n_batches, freeze=[], obj=['obs'], target_as_input=False):
		
		tolerance = 0.5 # loss < validation_loss + tol * std_validation_loss --> early stopping 
		chunker = Chunker(self.train_path,batch_size,chunk_size,self.chromatic,self.noise,self.dev)
  
		# Select which parameters to train
		parameters_to_train = []
		for name, parameter in self.net.named_parameters():
			if name.split('.')[0] in freeze:
				# print(f'Freezing {name}')
				parameter.requires_grad = False
			else:
				parameter.requires_grad = True
				parameters_to_train.append(parameter)

		optimizer = optim.Adam(parameters_to_train, lr=lr)
		loss_func = nn.BCELoss()
		loss_hist = []

		for batch in range(n_batches):
			
			optimizer.zero_grad()
			target, sample = chunker.generate_chunk()

			obs, pred = self.net(target if target_as_input else sample)

			loss = 0
			if 'obs' in obj:
				loss += loss_func(obs, target)
			if 'pred' in obj:
				loss += loss_func(pred[:, :-1], target[:, 1:])				

			loss.backward()
			optimizer.step()

			loss_hist += [float(loss.detach().cpu().numpy())]

			if batch % 100 == 0:

				# print(f"Batch {batch:04}/{n_batches} | Loss: {loss:.4f}")
				
				# Early stopping if we begin to overfit the data when training RNNs
				if 'rnn' not in freeze and batch > 0.05 * n_batches:
					
					ve = self.test(chunk_size, batch_size, 6, obj, self.validation_path)
					valid_error_m, valid_error_s = np.mean(ve), np.std(ve)
					cutoff = valid_error_m - tolerance * valid_error_s
					cutoff_str  = f'{valid_error_m:.2f}' + u'\u00B1' + f'{valid_error_s:.2f}'
					cutoff_str += f' (tol = {tolerance:.2f})'

					avg_loss = np.mean(loss_hist[max(0, batch-10):])
					
					if avg_loss < cutoff:
						print(f'Stopping -> <Loss> = {avg_loss:.2f}, val_loss = {cutoff_str}')
						break

		self._write_report_(loss_hist, freeze, obj)

	def test(self, chunk_size, batch_size, n_samples=1, obj=['obs'], test_path=None):

		if test_path is None:
			test_path = self.test_path
		chunker = Chunker(test_path, batch_size, chunk_size, self.chromatic, self.noise, self.dev)

		loss_func   = nn.BCELoss(reduction='none')
		performance = np.zeros((n_samples, batch_size))

		with torch.no_grad():
			
			for n_sample in (tqdm(range(n_samples)) if n_samples > 20  else range(n_samples)):	
				target, sample = chunker.generate_chunk()
				obs, pred = self.net(sample)

				loss = 0
				if 'obs' in obj:
					loss += loss_func(obs, target).mean((1,2))
				if 'pred' in obj:
					loss += loss_func(pred[:, :-1], target[:, 1:]).mean((1,2))

				performance[n_sample, :] = loss.cpu().numpy()		

		return performance

	def _write_report_(self, loss_hist, freeze, obj):

		report_name = self.modname
		report_name += f'_obj-{"-".join(obj)}_'
		report_name += f'freeze-{"none" if freeze == [] else "-".join(freeze)}'
		report_path = os.path.join(os.path.split(self.modpath)[0], report_name) + '.txt'
		history_str = ','.join([f'{l:.4f}' for l in loss_hist])

		with open(report_path, 'w') as f:
			f.write(history_str)

	def _compose_weightpath_(self, suffix):

		folder, fname = os.path.split(self.modpath)
		name, extension = '.'.join(fname.split('.')[:-1]), fname.split('.')[-1]
		filepath = os.path.join(folder, name + suffix + extension)

		return filepath


# Data Loading
class Chunker():

	def __init__(self, songs_path, batch_size, chunk_size, chromatic, noise, dev=None):
		
		if dev is None:
			dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.songs_path = songs_path
		self.batch_size = batch_size
		self.chunk_size = chunk_size
		self.chromatic  = chromatic
		self.noise      = noise
		self.dev        = dev

		self.songs_dict = self._get_songs_()
		self.song_pool  = list(self.songs_dict.keys())
		self.song_list = [np.random.choice(self.song_pool) for _ in range(batch_size)]
		self.t0  = [0 for _ in range(batch_size)]

	def generate_chunk(self):
		n_dims = self.songs_dict[self.song_pool[0]]['target'].shape[1]
		target = np.zeros((self.batch_size, self.chunk_size, n_dims))
		sample = np.zeros((self.batch_size, self.chunk_size, n_dims))

		for n in range(self.batch_size):
			t0, t1 = self.t0[n], self.t0[n] + self.chunk_size
			target[n, :, :] = self.songs_dict[self.song_list[n]]['target'][t0:t1]
			sample[n, :, :] = self.songs_dict[self.song_list[n]]['sample'][t0:t1]
			if t1 > self.songs_dict[self.song_list[n]]['target'].shape[1]-self.chunk_size:
				self.song_list[n] = np.random.choice(self.song_pool)
				self.t0[n] = 0
			else:
				self.t0[n] += self.chunk_size

		target = torch.tensor(target).to(device=self.dev).float()
		sample = torch.tensor(sample).to(device=self.dev).float()

		return target, sample

	def _get_songs_(self):
		
		n_pad_song_end = 20

		files = dict()
		for filepath in glob.glob(os.path.join(self.songs_path, '*.csv')):
			file = os.path.split(filepath)[-1]
			if '_' in file:
				opera = file.split('_')[0]
			else:
				opera = file.replace('.csv', '')
			files[opera] = glob.glob(os.path.join(self.songs_path, f'{opera}*.csv'))

		songs_dict = dict()
		for opera in files:
			songs_dict[opera] = dict()
			targets = []
			for file in files[opera]:
				with open(file, 'r') as f:
					targets += [el for el in list(csv.reader(f)) if el != []]
					targets += [[]] * n_pad_song_end

			target_array = self._generate_target_array_(targets)
			sample_array = self._add_noise_to_target_(target_array)
			songs_dict[opera]['target'] = target_array
			songs_dict[opera]['sample'] = sample_array

		return songs_dict

	def _generate_target_array_(self, targets):

		if self.chromatic:
			n_midi_notes = 12
		else:
			n_midi_notes   = 108
		
		song_len = int(self.chunk_size * np.ceil(len(targets)/self.chunk_size))
		target_array = np.zeros((song_len, n_midi_notes))

		for tick, target_chord in enumerate(targets):
			for target_note in [int(note)-1 for note in target_chord if note != '']:
				note = target_note % 12 if self.chromatic else target_note
				target_array[tick, note] = 1

		return target_array

	def _add_noise_to_target_(self, target_array):

		sample_array = target_array + self.noise * np.random.randn(*target_array.shape)

		return(sample_array)


# Data Retriever
class BWVRetriever():

	def __init__(self, save_dir, opera_list_file=None, verbose=False):

		os.makedirs(save_dir, exist_ok=True)

		if opera_list_file is None: 
			opera_list_file = os.path.join(save_dir, '_list.txt') 
			open(opera_list_file, 'w').close()
		
		self.save_dir        = save_dir
		self.opera_list_file = opera_list_file
		self.downloaded      = self.read_opera_list()
		self.verbose         = verbose



	def scrape_websites(self, base_urls=[]):

		if type(base_urls) is not list:
			base_urls = [base_urls]

		for base_url in base_urls:
			self._scrape_page_(base_url)

		self._write_opera_list_()


	def _scrape_page_(self, base_url, url=None, visited=None):

		if url     is None: url     = copy.copy(base_url)
		if visited is None: visited = set()
		
		if url in visited:
			return None

		visited.add(url)

		print(f'Scraping: {url}')
		response = requests.get(url)
		
		if response.status_code != 200:
			print(f'Failed to fetch {url}')
			return

		soup = BeautifulSoup(response.text, 'html.parser')

		for link in soup.find_all('a', href=True):
			full_url = urljoin(url, link['href'])

			# Ignore external links
			if base_url not in full_url:
				continue
			
			if full_url.lower().endswith(('.mid', '.midi', '.zip')): 
				opera, fname = self._match_bwv_(link.text)
				if opera is not None and opera not in self.downloaded:
					success = self._save_file_(full_url, fname) 
					if success: self.downloaded.append(opera)

			elif full_url.lower().endswith(('.html')): # Recursively scrape new pages
				self._scrape_page_(base_url, full_url, visited) 


	def _match_bwv_(self, link_text):

		# We are more lenient with MIDI files as they are guaranteed to be Bach's
		regex_midi   = r'^(?:[a-zA-Z_-]*)(\d{3,4}|(?<=bwv)\d{1,4})'
		regex_midi  += r'(?:[_\s-]*)?([^v\d][^v]*?)?(?:v\d+)?\.midi?$'
		rematch_midi = re.search(regex_midi, link_text.lower())

		# Zip files are forced to have a bwv in the name
		regex_zip    = r'^(?:.*bwv.*?)(\d{1,4}).*?\.zip$'
		rematch_zip  = re.search(regex_zip, link_text.lower())

		if rematch_midi:
			opera  = f'bwv{int(rematch_midi.group(1)):04d}'
			suffix = rematch_midi.group(2)
			
			if suffix is None: suffix = ''
			else: suffix = f"_{re.sub(r'[^a-zA-Z0-9]', '', suffix)}"
		
			fname = opera + suffix + '.mid'
		
		elif rematch_zip:
			opera = f'bwv{int(rematch_zip.group(1)):04d}'
			fname = opera + '.zip'

		else:
			opera, fname = None, None

		if self.verbose: print(f'{link_text} → {fname}')
		
		return opera, fname


	def _save_file_(self, file_url, fname):
			
		if self.verbose: print(f'saving: {file_url}')
		response = requests.get(file_url, stream=True)
		
		if response.status_code == 200:

			if 'mid' in os.path.splitext(fname)[1]:
				
				tmp_filepath = os.path.join(self.save_dir, 'tmp.mid')
				with open(tmp_filepath, 'wb') as f:
					for chunk in response.iter_content(chunk_size=1024):
						f.write(chunk)
				
				targets = self._read_targets_from_midi_(tmp_filepath)
				os.remove(tmp_filepath)

				file_name = os.path.join(self.save_dir, os.path.splitext(fname)[0]+'.csv')
				with open(file_name, 'w') as f:
					csv.writer(f).writerows(targets)


			elif os.path.splitext(fname)[1] == 'zip':
				pass 

				# ToDo: If zip files with .mid files exist and the files contain 
				# works that are not listed in the global bwv list, unzip the file
				# and save the relevant midis
				"""
			    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
			        for zip_info in zip_ref.infolist():
			            if zip_info.filename.lower().endswith(('.mid', '.midi')):  # Check for MIDI files
			                # Rename file: remove .zip from fname and append MIDI file name
			                zip_base = os.path.splitext(fname)[0]  # Remove .zip extension
			                new_fname = f"{zip_base}_{os.path.basename(zip_info.filename)}"
			                extract_path = os.path.join(save_dir, new_fname)

			                # Extract and save file directly
			                with zip_ref.open(zip_info.filename) as source, open(extract_path, 'wb') as target:
			                    target.write(source.read())
			    """

		else:
			if self.verbose: print(f'Failed to save: {file_url}')

		return response.status_code == 200


	def _read_targets_from_midi_(self, midi_file):

		# 1) extract notes from midi using mido
		mid = mido.MidiFile(midi_file, clip=True)

		notes = []
		pending = {}
		for track in mid.tracks: 
			abstime = 0 # reset absolute time at the start of each track.
			for el in track:
				abstime += el.time
				if (el.type=='note_on' and el.velocity>0):
					pending[el.note] = {'note': el.note, 't0': abstime}
				if (el.type=='note_off' or (el.type=='note_on' and el.velocity==0)):
					if el.note in pending:
						this_note = pending[el.note]
						this_note['t1'] = abstime
						notes.append(this_note)
						del pending[el.note]

		# 2) parcelate streams into single-chord boxes 
		mbox = np.nan * np.ones((1, max([note['t1'] for note in notes])))
		for ix, note in enumerate(notes):
			t0, t1, n = note['t0'], note['t1'], 0
			while not np.isnan(mbox[n, t0:t1]).all():
				n += 1
				if n >= mbox.shape[0]:
					mbox = np.pad(mbox, ((0,1),(0,0)), constant_values=np.nan)
			mbox[n, t0:t1] = note['note']

		# 3) locate the targets within teach tick of the music box
		target_list = [[int(note) for note in mbox[:, 0] if not np.isnan(note)]]
		for tick in range(1, mbox.shape[1]):
			this_target = [int(note) for note in mbox[:, tick] if not np.isnan(note)]
			if this_target != target_list[-1] and not this_target == []:
				target_list.append(this_target)

		return target_list


	def read_opera_list(self):

		with open(self.opera_list_file, 'r') as f:
			opera_list = f.read().split('\n')

		opera_list = [opera for opera in opera_list if opera != '']

		return sorted(opera_list)


	def _write_opera_list_(self):

		opera_list = [opera for opera in self.downloaded if opera != '']

		with open(self.opera_list_file, 'w') as f:
			f.write('\n'.join(sorted(opera_list)))




def test_globaldist_model(pars, n_train_samples, n_test_samples, dev=None):

	if dev is None:
		dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if n_train_samples > 512:
		n_batches  = int(n_train_samples/512)
		batch_size = 512
	else:
		n_batches = 1
		batch_size = n_train_samples

	train_chunker = Chunker(os.path.join(pars['datapath'], 'train'),
							batch_size,
							pars['chunk_size'],
							pars['chromatic'], 
							pars['noise'],
							dev)

	test_chunker  = Chunker(os.path.join(pars['datapath'], 'test'), 
							n_test_samples, 
							pars['chunk_size'],
							pars['chromatic'],
							pars['noise'],
							dev)

	glob_dist = torch.zeros(12 if pars['chromatic'] else 108).to(dev)
	for n in range(n_batches):
		glob_dist += train_chunker.generate_chunk()[1].mean((0, 1))
	glob_dist *= 1/glob_dist.sum()

	targets     = test_chunker.generate_chunk()[0]
	loss_func   = nn.BCELoss(reduction='none')
	performance = loss_func(glob_dist.expand(targets.shape), targets)

	return performance






