import csv
import numpy as np
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

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
		self.train_path      = os.path.join(pars['datapath'], 'train')
		self.validation_path = os.path.join(pars['datapath'], 'test')
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






