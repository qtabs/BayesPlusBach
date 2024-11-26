import csv
import mido
import numpy as np

def import_target_list_from_csv(filename):

	with open(filename, 'r') as f:
		target_list = list(csv.reader(f))

	target_list = [[int(t) for t in target] for target in target_list]

	return target_list


def generate_target_array(target_list):

	n_midi_notes   = 108
	n_pad_song_end = 20

	# Torch RNN input (batch_first = True): (n_batches, seq_len, n_dimensions)
	target = np.zeros((1, len(target_list) + n_pad_song_end, n_midi_notes))
	
	for token_number, token in enumerate(target_list):
		for note_number in token:
			target[0, token_number, note_number - 1] = 1

	return target


def generate_observation(target, noise=0.5):

	observation = target + 0.5 * np.random.randn(*target.shape)

	return(observation)


midi_filename = './bwv0539.mid'
csv_filename  = midi_filename.replace('.mid', '.csv')

target_list = import_target_list_from_csv(csv_filename)
target = generate_target_array(target_list)
observation = generate_observation(target, 0.5)
