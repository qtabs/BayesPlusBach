import csv
import mido
import numpy as np

def read_notes(midi_file):
	""" Reads midi_file and returns a list of dictionaries with fields:
		t0:   note starting time
		t1:   note ending time
		note: number of the note 
	"""

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

	return notes


def compute_music_box(notes):

	mbox = np.nan * np.ones((1, max([note['t1'] for note in notes])))
	for ix, note in enumerate(notes):
		t0, t1, n = note['t0'], note['t1'], 0
		while not np.isnan(mbox[n, t0:t1]).all():
			n += 1
			if n >= mbox.shape[0]:
				mbox = np.pad(mbox, ((0,1),(0,0)), constant_values=np.nan)
		mbox[n, t0:t1] = note['note']

	return mbox


def generate_target_list(mbox):

	target_list = [[int(note) for note in mbox[:, 0] if not np.isnan(note)]]
	for tick in range(1, mbox.shape[1]):
		this_target = [int(note) for note in mbox[:, tick] if not np.isnan(note)]
		if this_target != target_list[-1] and not this_target == []:
			target_list.append(this_target)

	return target_list


def export_target_list_to_csv(target_list, filename):

	with open(filename, 'w') as f:
		csv.writer(f).writerows(target_list)


midi_filename = './bwv0539.mid'
csv_filename  = midi_filename.replace('.mid', '.csv')

notes  = read_notes(sample_MIDI)
mbox   = compute_music_box(notes)
target_list = generate_target_list(mbox)
export_target_list_to_csv(target_list, csv_filename)

