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
from urllib.parse import urljoin, urlparse, parse_qs


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

		return x_obs, x_pred, x


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
		# x serves as either inferred value or prediction
		return x, x, None


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
		w = torch.load(weightpath, map_location=self.dev, weights_only=True)
		self.net.load_state_dict(w)

	def save_weights(self, suffix=''):
		weightpath = self._compose_weightpath_(suffix)
		torch.save(self.net.state_dict(), weightpath)

	def trained_weights_exists(self, suffix=''):
		weightpath = self._compose_weightpath_(suffix)
		return os.path.exists(weightpath)

	def train(self, lr, chunk_size, batch_size, n_batches, freeze=[], obj=['obs'], target_as_input=False):
		
		tolerance = 1 # loss < validation_loss + tol * std_validation_loss --> early stopping 
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

			obs, pred, _ = self.net(target if target_as_input else sample)

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
				if 'rnn' not in freeze and batch > 0.10 * n_batches:
					
					ve = self.test(obj, test_path=self.validation_path)
					valid_error_m, valid_error_s = np.mean(ve), np.std(ve)
					cutoff = valid_error_m - tolerance * valid_error_s
					cutoff_str  = f'{valid_error_m:.2f}' + u'\u00B1' + f'{valid_error_s:.2f}'
					cutoff_str += f' (tol = {tolerance:.2f})'

					avg_loss = np.mean(loss_hist[max(0, batch-10):])
					
					if avg_loss < cutoff:
						print(f' [Early stopping with loss{avg_loss:.2f} < {cutoff_str}] ', end='')
						break

		self._write_report_(loss_hist, freeze, obj)

	def test(self, obj=['obs'], test_path=None):

		if test_path is None:
			test_path = self.test_path

		with torch.no_grad():
		
			chunker = Chunker(test_path, 1, None, self.chromatic, self.noise, self.dev)
			operas  = sorted(chunker.song_pool)

			performance = np.empty(len(operas))
			loss_func   = nn.BCELoss(reduction='none')

			for n, opera in enumerate(operas):
				
				target = chunker.read_song_as_tensor(opera)['target']
				sample = chunker.read_song_as_tensor(opera)['sample']

				obs, pred, _   = self.net(sample)

				if 'obs' in obj:
					performance[n] = loss_func(obs, target).mean().cpu().numpy()
				if 'pred' in obj:
					performance[n] = loss_func(pred[:, :-1], target[:, 1:]).mean().cpu().numpy()

		return performance

	def compute_pe_and_state(self, sample):

		with torch.no_grad():

			obs, pred, hidden = self.net(sample)
			pred   = pred.cpu().detach().numpy()
			obs    = obs.cpu().detach().numpy()
			hidden = hidden.cpu().detach().numpy()
			sample = sample.cpu().detach().numpy()

		pe  = sample[0, 1:] - pred[0, :-1]
		stm = hidden[0, 1:]
		std = hidden[0, 1:] - hidden[0, :-1]

		return pe, stm, std

	def test_globaldist_model(self):
		
		with torch.no_grad():
			# Compute global distribution from training set
			train_chunker = Chunker(self.train_path, 1, None, self.chromatic, self.noise, self.dev)
			train_operas  = sorted(train_chunker.song_pool)

			glob_dist = torch.zeros(self.net.n_dim).to(self.dev)
			
			for opera in train_operas:
				glob_dist += train_chunker.read_song_as_tensor(opera)['sample'].mean((0, 1))
			
			glob_dist *= 1/glob_dist.sum()

		# Measure performance in the testing set
		with torch.no_grad():
			test_chunker = Chunker(self.test_path, 1, None, self.chromatic, self.noise, self.dev)
			test_operas  = sorted(test_chunker.song_pool)

			performance = np.empty(len(test_operas))
			loss_func   = nn.BCELoss(reduction='none')
			
			for n, opera in enumerate(test_operas):
				target = test_chunker.read_song_as_tensor(opera)['target']
				performance[n] = loss_func(glob_dist.expand(target.shape), target).mean().cpu().numpy()

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
		filepath = os.path.join(folder, name + suffix + '.' + extension)

		return filepath


# Data Loading
class Chunker():

	def __init__(self, songs_path, batch_size, chunk_size, chromatic, noise, dev=None):
		
		if dev is None:
			dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.song_end_pad = 20

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

	def read_song_as_tensor(self, opera):
	
		song = self.songs_dict[opera]

		real_len = song['target'].shape[0]
		for n_notes in song['target'].sum(1)[::-1]:
			if n_notes == 0:
				real_len -= 1
			else:
				break

		options = {'dtype': torch.float32, 'device': self.dev}
		song_tensor = {k: torch.tensor(song[k][None,:real_len], **options) for k in song}

		return song_tensor

	def _get_songs_(self):
		
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
					targets += [[]] * self.song_end_pad

			target_array = self._generate_target_array_(targets)
			sample_array = self._add_noise_to_target_(target_array)
			songs_dict[opera]['target'] = target_array
			songs_dict[opera]['sample'] = sample_array

		return songs_dict

	def _generate_target_array_(self, targets):

		n_midi_notes = 12 if self.chromatic else 108
			
		if self.chunk_size is None:
			song_len = len(targets)
		else:
			max_chunks = np.ceil(len(targets) / self.chunk_size)
			song_len   = int(self.chunk_size * max_chunks)
		
		target_array = np.zeros((song_len, n_midi_notes))

		for tick, target_chord in enumerate(targets):
			notes = [int(note)-1 for note in target_chord if note != '']
			for target_note in notes:
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

	def delete_duplicates(self):

		files_by_prefix = {}

		for filename in os.listdir(self.save_dir):
		    if filename.endswith('.csv') and '_' in filename:
		        prefix = filename.split('_')[0]
		        files_by_prefix.setdefault(prefix, []).append(filename)

		for prefix, files in files_by_prefix.items():
		    lengths = {}
		    for f in files:
		        file_path = os.path.join(self.save_dir, f)
		        try:
		            with open(file_path, 'r') as file:
		                length = sum(1 for line in file) - 1

		        except Exception as e:
		            print(f"Error reading {f}: {e}")
		            continue

		        if length in lengths:
		            os.remove(file_path)
		            print(f"Deleted duplicate: {f} (same length as {lengths[length]})")
		        else:
		            lengths[length] = f
	
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

		# Build filename → BWV mapping from the table (only if suzumidi)
		filename_to_bwv = {}
		current_bwv_range = None

		is_suzumidi = 'suzumidi' in url.lower()

		if is_suzumidi:
			for row in soup.find_all('tr'):
				cells = row.find_all('td')
				if len(cells) >= 2:
					title_text = cells[0].get_text(strip=True)

					# Check for BWV range
					range_match = re.search(r'BWV\s?(\d{3,4})-(\d{3,4})', title_text, re.IGNORECASE)
					if range_match:
						start_num = int(range_match.group(1))
						end_num = int(range_match.group(2))
						current_bwv_range = (start_num, end_num)
					else:
						current_bwv_range = None

					file_link = cells[1].find('a', href=True)
					if file_link:
						filename = file_link['href'].split('/')[-1]

						if current_bwv_range:
							# Take BWV number from filename
							bwv_num_match = re.search(r'(\d{3,4})', filename.lower())
							if bwv_num_match:
								num = int(bwv_num_match.group(1))
								if current_bwv_range[0] <= num <= current_bwv_range[1]:
									bwv_num = f"bwv{num:04d}"
									filename_to_bwv[filename.lower()] = bwv_num
									continue

						# Single BWV from table title
						bwv_match = re.search(r'BWV\s?(\d{3,4})', title_text, re.IGNORECASE)
						if bwv_match:
							bwv_num = f"bwv{int(bwv_match.group(1)):04d}"
							filename_to_bwv[filename.lower()] = bwv_num

		# Main loop over links
		base_domain = urlparse(base_url).netloc
		allowed_paths = ['/midi/', '/music/', '/works/', '/midi-list/', '/bach-ken/',
						'/bwv-list/', '/musique/', '/prs/']

		for link in soup.find_all('a', href=True):
			full_url = urljoin(url, link['href'])
			parsed = urlparse(full_url)

			url_path = parsed.path.lower()
			url_query = parsed.query
			full_domain = parsed.netloc

			if full_domain != base_domain:
				continue

			downloaded_this_round = False

			# Case 1: Direct file
			if url_path.endswith(('.mid', '.midi', '.zip')):
				filename = url_path.split('/')[-1]

				opera = filename_to_bwv.get(filename.lower()) if is_suzumidi else None

				if opera is None:
					opera, fname = self._match_bwv_(filename)
				else:
					# For suzumidi: special single BWV case
					if filename.lower().endswith(('.mid', '.midi')):
						suffix_match = re.search(r'(\d{3,4})([^a-zA-Z0-9]?)(.*)\.mid', filename.lower())
						if suffix_match and current_bwv_range is None:
							suffix = re.sub(r'[^a-zA-Z0-9]', '', suffix_match.group(3))
							fname = opera + (f'_{suffix}' if suffix else '') + '.mid'
						else:
							fname = opera + '.mid'
					elif filename.lower().endswith('.zip'):
						fname = opera + '.zip'
					else:
						fname = filename

				if opera and opera not in self.downloaded:
					if self._save_file_(full_url, fname, opera=opera, suzumidi=is_suzumidi):
						self.downloaded.append(opera)
						downloaded_this_round = True

			# Case 2: Query string
			if not downloaded_this_round and url_query:
				qs = parse_qs(url_query)
				if 'file' in qs:
					for f in qs['file']:
						if f.lower().endswith(('.mid', '.midi', '.zip')):
							filename = f.split('/')[-1]
							opera = filename_to_bwv.get(filename.lower()) if is_suzumidi else None

							if opera is None:
								opera, fname = self._match_bwv_(filename)
							else:
								if filename.lower().endswith(('.mid', '.midi')):
									fname = opera + '.mid'
								elif filename.lower().endswith('.zip'):
									fname = opera + '.zip'
								else:
									fname = filename

							if opera and opera not in self.downloaded:
								if self._save_file_(full_url, fname, opera=opera, suzumidi=is_suzumidi):
									self.downloaded.append(opera)
									downloaded_this_round = True

			# Recursively scrape
			if (url_path.endswith(('.html', '.php')) or url_query) and 'bach' in full_url.lower():
				if any(path in url_path for path in allowed_paths):
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

	def _save_file_(self, file_url, fname, opera=None, suzumidi=False, filename_to_bwv=None):
	
		if self.verbose:
			print(f'saving: {file_url}')
		
		response = requests.get(file_url, stream=True)

		if response.status_code == 200:

			if os.path.splitext(fname)[1] in ('.mid', '.midi'):
				
				tmp_filepath = os.path.join(self.save_dir, 'tmp.mid')
				with open(tmp_filepath, 'wb') as f:
					for chunk in response.iter_content(chunk_size=1024):
						f.write(chunk)

				try:
					targets = self._read_targets_from_midi_(tmp_filepath)
				except OSError as e:
					if self.verbose:
						print(f"Failed to parse MIDI file {fname} from {file_url}: {e}")
					os.remove(tmp_filepath)
					return False

				os.remove(tmp_filepath)

				if targets is None:
					if self.verbose:
						print(f"Skipping file {file_url} due to failed MIDI parsing.")
					return False

				file_name = os.path.join(self.save_dir, os.path.splitext(fname)[0] + '.csv')
				with open(file_name, 'w') as f:
					csv.writer(f).writerows(targets)

			elif os.path.splitext(fname)[1] == '.zip' and suzumidi:
				
				with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
					midi_count = 0

					for zip_info in zip_ref.infolist():
						if zip_info.filename.lower().endswith(('.mid', '.midi')):
							midi_count += 1
							midi_filename = os.path.basename(zip_info.filename).lower()

							# Extract 3-4 digit number from the filename inside ZIP
							number_match = re.search(r'(\d{3,4})', midi_filename)
							if number_match:
								num = int(number_match.group(1))
								new_fname_base = f'bwv{num:04d}'
							else:
								# fallback name
								zip_base = opera if opera else os.path.splitext(fname)[0]
								suffix_match = re.search(r'([a-zA-Z0-9]+)\.mid$', midi_filename)
								suffix = suffix_match.group(1) if suffix_match else f"{midi_count}"
								new_fname_base = f"{zip_base}_{suffix}"

							with zip_ref.open(zip_info.filename) as source:
								
								midi_bytes = source.read()
								tmp_filepath = os.path.join(self.save_dir, 'tmp.mid')
								
								with open(tmp_filepath, 'wb') as f:
									f.write(midi_bytes)

								try:
									targets = self._read_targets_from_midi_(tmp_filepath)
								except OSError as e:
									if self.verbose:
										print(f"Skipping file inside ZIP due to failed MIDI parsing: {midi_filename}: {e}")
									os.remove(tmp_filepath)
									continue

								os.remove(tmp_filepath)

								if targets is None:
									if self.verbose:
										print(f"Skipping file inside ZIP due to failed MIDI parsing: {midi_filename}")
									continue

								file_name = os.path.join(self.save_dir, new_fname_base + '.csv')
								with open(file_name, 'w') as f:
									csv.writer(f).writerows(targets)

								if self.verbose:
									print(f"Extracted and saved: {file_name}")

			else:
				# fallback for other file types
				if os.path.splitext(fname)[1].lower() == '.zip' and not suzumidi:
					if self.verbose:
						print(f"Skipping saving ZIP file {fname} because suzumidi=False")
				else:
					with open(os.path.join(self.save_dir, fname), 'wb') as f:
						f.write(response.content)

		else:
			if self.verbose:
				print(f'Failed to save: {file_url}')

		return response.status_code == 200

	def _read_targets_from_midi_(self, midi_file):

		# 1) extract notes from midi using mido
		try:
			mid =  mido.MidiFile(midi_file, clip=True)
		except mido.midifiles.meta.KeySignatureError as e:
			print(f"Skipping file due to KeySignatureError: {midi_file} ({e})")
			return None

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



############



