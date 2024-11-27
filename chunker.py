import numpy as np

class Chunker():
	def __init__(self, songs_dict, n_batches, chunk_size):
		self.n_batches = n_batches
		self.songs_dict = songs_dict
		self.chunk_size = chunk_size
		self.song_pool  = list(songs_dict.keys())
		self.song_list = [np.random.choice(self.song_pool) for _ in range(n_batches)]
		self.t0  = [0 for _ in range(n_batches)]

	def create_chunck(self):
		n_dims = self.songs_dict[self.song_pool[0]]['target'].shape[2]
		target = np.zeros((self.n_batches, self.chunk_size, n_dims))
		sample = np.zeros((self.n_batches, self.chunk_size, n_dims))

		for n in range(self.n_batches):
			t0, t1 = self.t0[n], self.t0[n] + self.chunk_size
			target[n, :, :] = self.songs_dict[self.song_list[n]]['target'][0, t0:t1]
			sample[n, :, :] = self.songs_dict[self.song_list[n]]['sample'][0, t0:t1]
			if t1 == self.songs_dict[self.song_list[n]]['target'].shape[1]:
				self.song_list[n] = np.random.choice(self.song_pool)
				self.t0[n] = 0
			else:
				self.t0[n] += self.chunk_size

		return target, sample

example_dict = dict()
for n, m in zip(range(5), [3,4,1,5]):
	example_dict[f'bwv{n:04d}'] = {'target': np.random.rand(1, 16*m, 12), 'sample': np.random.rand(1, 512*m, 12)}

chunker = Chunker(example_dict, 4, 16)
target, sample = chunker.create_chunck()
print(target.shape)
print(sample.shape)
