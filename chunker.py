import csv
import mido
import numpy as np
import glob
import os


def generate_target_array(targets, chunk_size, chromatic=True):

	if chromatic:
		n_midi_notes = 12
	else:
		n_midi_notes   = 108
	
	song_len = int(chunk_size * np.ceil(len(targets)/chunk_size))
	target_array = np.zeros((song_len, n_midi_notes))

	for tick, target_chord in enumerate(targets):
		for target_note in [int(note)-1 for note in target_chord if note != '']:
			note = target_note % 12 if chromatic else target_note
			target_array[tick, note] = 1

	return target_array


def add_noise_to_target(target_array, noise=0.5):

	sample_array = target_array + 0.5 * np.random.randn(*target_array.shape)

	return(sample_array)


def get_songs(songs_path, chunk_size, chromatic=True, noise=0.5):
	
	n_pad_song_end = 20

	files = dict()
	for filepath in glob.glob(os.path.join(songs_path, '*.csv')):
		file = os.path.split(filepath)[-1]
		if '_' in file:
			opera = file.split('_')[0]
		else:
			opera = file.replace('.csv', '')
		files[opera] = glob.glob(os.path.join(songs_path, f'{opera}*.csv'))

	songs_dict = dict()
	for opera in files:
		songs_dict[opera] = dict()
		targets = []
		for file in files[opera]:
			with open(file, 'r') as f:
				targets += [el for el in list(csv.reader(f)) if el != []]
				targets += [[]] * n_pad_song_end

		target_array = generate_target_array(targets, chunk_size, chromatic)
		sample_array = add_noise_to_target(target_array, noise)
		songs_dict[opera]['target'] = target_array
		songs_dict[opera]['sample'] = sample_array

	return songs_dict


class Chunker():

	def __init__(self, songs_path, batch_size, chunk_size, chromatic=True, noise=0.5):
		self.batch_size = batch_size
		self.songs_dict = get_songs(songs_path, chunk_size, chromatic, noise)
		self.chunk_size = chunk_size
		self.song_pool  = list(self.songs_dict.keys())
		self.song_list = [np.random.choice(self.song_pool) for _ in range(batch_size)]
		self.t0  = [0 for _ in range(batch_size)]

	def create_chunck(self):
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

		return target, sample


chunk_size = 512
batch_size = 16
chromatic  = True
noise = 0.5

chunker = Chunker('./bach_CSV', chunk_size, batch_size, chromatic, noise)
target, sample = chunker.create_chunck()
print(target.shape)
print(sample.shape)
