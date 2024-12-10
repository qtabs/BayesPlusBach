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

	def __init__(self, n_dim, n_layers, n_hidden, unit_type, batch_size=None):
		
		super(RNN, self).__init__()

		self.n_dim     = n_dim
		self.n_layers  = n_layers
		self.n_hidden  = n_hidden
		self.unit_type = unit_type

		if self.unit_type.lower() == 'elman':
			self.rnn = nn.RNN(n_dim, n_hidden, n_layers, batch_first=True)
		elif self.unit_type.lower() == 'gru':
			self.rnn = nn.GRU(n_dim, n_hidden, n_layers, batch_first=True)
		elif self.unit_type.lower() == 'lstm':
			self.rnn = nn.LSTM(n_dim, n_hidden, n_layers, batch_first=True)
		else:
			print('Unit type {self.unit_type} not recognised')

		self.out_obs  = nn.Linear(n_hidden, n_dim, bias=True)
		self.sig_obs  = nn.Sigmoid()
		self.out_pred = nn.Linear(n_hidden, n_dim, bias=True)
		self.sig_pred = nn.Sigmoid()

		if batch_size is not None:
			self.init_hidden(batch_size)
		else:
			self.hidden = None

	def init_hidden(self, batch_size):
		h_size = [self.n_layers, batch_size, self.n_hidden]
		self.hidden = torch.zeros(*h_size)

	def forward(self, x):

		self.init_hidden(x.shape[0])
		x, self.hidden = self.rnn(x, self.hidden)
		x_obs  = self.sig_obs(self.out_obs(x))
		x_pred = self.sig_pred(self.out_pred(x))

		return x_obs, x_pred


class FeedForwardNN(nn.Module):

	def __init__(self, n_dim, n_hidden_layers, n_hidden_units):
		
		super(FeedForwardNN, self).__init__()

		self.n_dim           = n_dim
		self.n_hidden_layers = n_hidden_layers
		self.n_hidden_units  = n_hidden_units

		layers = [nn.Linear(self.n_dim, self.n_hidden_units), nn.ReLU()]
		for _ in range(self.n_hidden_layers):
			layers.append(nn.Linear(self.n_hidden_units, self.n_hidden_units))
			layers.append(nn.ReLU())
		layers += [nn.Linear(self.n_hidden_units, self.n_dim), nn.Sigmoid()]

		self.feedforward = nn.Sequential(*layers)

	def forward(self, x):
		x = self.feedforward(x)  # value OR probability???????
		return x


# Data Loading
class Chunker():

	def __init__(self, songs_path, batch_size, chunk_size, chromatic, noise, dev):
		
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
		self.chromatic  = pars['chromatic']
		self.noise      = pars['noise']
		self.train_path = os.path.join(pars['datapath'] , 'train')
		self.test_path  = os.path.join(pars['datapath'] , 'test')

		n_dim = 12 if self.chromatic else 108
		n_layers  = pars['n_layers']
		n_hidden  = pars['n_hidden']
		unit_type = pars['unit_type']

		self.rnn = RNN(n_dim, n_layers, n_hidden, unit_type).to(dev)

		modname =  f'{unit_type}_nhidden-{n_hidden:d}_nlayers-{n_layers:d}'
		modname += f'{"_chromatic" if self.chromatic else ""}'
		modname += f'_noise{self.noise}'.replace('.', 'p')
		self.modname = modname
		self.modpath = f'./models/{modname}.pth'

	def load_weights(self, suffix=''):
		modpath  = self.modpath.split('.')
		loadpath = modpath[0] + suffix + modpath[1]
		self.rnn.load_state_dict(torch.load(loadpath))

	def save_weights(self, suffix=''):
		modpath = self.modpath.split('.')
		savepath = modpath[0] + suffix + modpath[1]
		torch.save(self.rnn.state_dict(), self.modpath)

	def train(self, lr, chunk_size, batch_size, n_batches, freeze=[], obj=['obs']):
		
		chunker = Chunker(self.train_path, batch_size, chunk_size, 
						  self.chromatic, self.noise, self.dev)

		# Select which parameters to train
		parameters_to_train = []
		for name, parameter in self.rnn.named_parameters():
			if name.split('.')[0] in freeze:
				print(f'Freezing {name}')
				parameter.requires_grad = False
			else:
				parameter.requires_grad = True
				parameters_to_train.append(parameter)

		optimizer = optim.Adam(parameters_to_train, lr=lr)
		loss_func = nn.BCELoss()

		for batch in range(n_batches):
			
			optimizer.zero_grad()
			target, sample = chunker.generate_chunk()

			obs, pred = self.rnn(sample)

			loss = 0
			if 'obs' in obj:
				loss += loss_func(obs, target)
			if 'pred' in obj:
				loss += loss_func(pred[:, :-1], target[:, 1:])				

			loss.backward()
			optimizer.step()

			if batch % 10 == 0:
				print(f"Batch {batch+1:02}/{n_batches} |  Loss: {loss:.4f}")

	def test(self, chunk_size, batch_size, n_samples, obj=['obs']):

		chunker = Chunker(self.test_path, batch_size, chunk_size, 
						  self.chromatic, self.noise, self.dev)

		loss_func   = nn.BCELoss()
		performance = np.zeros(n_samples)

		with torch.no_grad():
			
			for n_sample in tqdm(range(n_samples)):	
				target, sample = chunker.generate_chunk()
				obs, pred = self.rnn(sample)

				loss = 0
				if 'obs' in obj:
					loss += loss_func(obs, target)
				if 'pred' in obj:
					loss += loss_func(pred[:, :-1], target[:, 1:])

				performance[n_sample] = loss.cpu().numpy()				

		return performance


##############



