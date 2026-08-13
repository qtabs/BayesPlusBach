import os
import bachbayes
import glob
import numpy as np
import os
import pickle
import random
import shutil
import time
import torch
import scipy.stats as ss
from sklearn.linear_model import RidgeCV

## Main analysis
NOISE_VALS    = [0.001, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0]
N_HIDDEN_VALS = [8, 16, 32, 64, 128, 256]
N_RUNS        = 10   # storage capacity of the results arrays
N_RUNS_TRAIN  = 5    # runs trained by default (extendable up to N_RUNS for free:
N_REF         = 256  # reference models are capacity-saturated (max of the sweep)

## PE analysis
# 'raw' == innovation; 'sig' == signal component; 'neg' == omission component
PE_TARGETS  = ['raw', 'sig', 'neg']
# Control ladder: 'b' == constant decoder; 'x' == observation; 'h' == hidden state
PE_DECODERS = ['b', 'x', 'h']


def prepare_data(test_rat=0.2, validation_rat=0.1, basedir='.'):

	base_urls = ["http://www.jsbach.net/midi/", 
				 "https://www.metronimo.com/fr/midi/index.php?page=Bach", 
				 "http://piano-midi.de/bach.htm",
				 "https://sound.jp/bach-ken/en/midi-list.html",
				 "http://www.dardel.info/musique/Bach.html",
				 "https://www.classicalarchives.com/prs/free.html",
				 "https://suzumidi.com/en/bach1.htm",
				 "https://suzumidi.com/en/bach2.htm",
				 "https://suzumidi.com/en/bach3.htm",
				 "https://suzumidi.com/en/bach4.htm",
				 "https://suzumidi.com/en/bach5.htm",
				 "https://suzumidi.com/en/bach6.htm"]
	save_dir  = 'data'

	retriever = bachbayes.BWVRetriever(save_dir, verbose=True)
	retriever.scrape_websites(base_urls)
	retriever.delete_duplicates()

	split_data(test_rat, validation_rat, basedir, save_dir)


def split_data(test_rat=0.2, validation_rat=0.1, basedir='.', save_dir='data'):

	# Group files by BWV number so that every version/movement of a composition
	# lands in the same split (splitting by file leaks pieces across sets)
	groups = dict()
	for fpath in glob.glob(os.path.join(save_dir, 'bwv*.csv')):
		fname = os.path.split(fpath)[1]
		opera = fname.split('_')[0] if '_' in fname else fname.replace('.csv', '')
		groups.setdefault(opera, []).append(fname)

	operas = sorted(groups)
	random.shuffle(operas)

	n_test_samples  = int(np.round(len(operas) * test_rat))
	n_val_samples   = int(np.round(len(operas) * validation_rat))

	samples = {'test': operas[:n_test_samples],
			   'validation': operas[n_test_samples:(n_val_samples+n_test_samples)],
			   'training': operas[(n_val_samples+n_test_samples):]}

	for key in samples:
		sample_dir = os.path.join(basedir, key)
		if os.path.exists(sample_dir):
			shutil.rmtree(sample_dir)  # clear files from any previous split
		os.makedirs(sample_dir)
		for opera in samples[key]:
			for fname in groups[opera]:
				shutil.copy(os.path.join(save_dir, fname), os.path.join(sample_dir, fname))


def main_fitting(noise, n_hidden, run_n=0, only_testing=False):

	# Parameters
	lr = 0.02
	chunk_size = 512
	batch_size = 512
	max_batches_obs = 20000
	n_batches_pred  = 3000

	pars = {'datapath'   : '.',
			'chunk_size' : 512,
			'chromatic'  : True,
			'noise'      : noise,
			'n_layers'   : 1,
			'n_hidden'   : n_hidden,
			'mod_type'   : 'gru'}

	m = bachbayes.Bachmodel(pars)


	def fit_or_load(weight_suffix, n_batches, **train_kwargs):

		# Resume at stage granularity.
		# The report suffix is the run alone: _write_report_ already
		# disambiguates the two stages by objective and freeze list
		if m.trained_weights_exists(weight_suffix):
			m.load_weights(weight_suffix)
			print('   already trained; loaded from disk')
			return

		t0 = time.time()
		m.train(lr, chunk_size, batch_size, n_batches, suffix=f'_run{run_n:02d}', **train_kwargs)
		m.save_weights(weight_suffix)
		print(f' done! Time = {(time.time() - t0)/60:.1f} minutes')


	# Training
	if only_testing:
		m.load_weights(f'_prediction_run{run_n:02d}')

	else:
		## Training RNN on observation accuracy
		print(f'Training on observations ({max_batches_obs} batches)...')
		fit_or_load(f'_observation_run{run_n:02d}', max_batches_obs,
					freeze=['out_pred'], obj=['obs'])

		## Training linear readout on prediction accuracy
		print(f'Training on predictions ({n_batches_pred} batches)...')
		fit_or_load(f'_prediction_run{run_n:02d}', n_batches_pred,
					freeze=['rnn', 'out_obs'], obj=['pred'])

	# Testing
	obs_error  = m.test(obj=['obs'])
	pred_error = m.test(obj=['pred'])

	# Reporting
	obs_err_m,  obs_err_e  = obs_error.mean(),  obs_error.std()
	pred_err_m, pred_err_e = pred_error.mean(), pred_error.std()

	print('\n---------------------------------')
	print(f'| Observation error = {obs_err_m:.2f}' + u'\u00B1' + f'{obs_err_e:.2f} |')
	print(f'| Prediction error  = {pred_err_m:.2f}' + u'\u00B1' + f'{pred_err_e:.2f} |')
	print('---------------------------------\n')

	performance = {'obs_err': obs_error, 'pred_err': pred_error}

	return performance


def baselines(noise, n_hidden=N_REF, only_testing=False):

	# Reference models estimate properties of the task (conditional entropies)

	# Parameters
	lr = 0.02
	chunk_size = 512
	batch_size = 512
	max_batches_rnn = 25000

	base_pars = {'datapath'   : '.',
				 'chunk_size' : chunk_size,
				 'chromatic'  : True,
				 'noise'      : noise,
				 'n_hidden'   : n_hidden}

	def logistic_ref(name, suffix, obj, target_as_input=False):

		m = bachbayes.LogisticModel({'datapath': '.', 'chromatic': True, 'noise': noise})
		print(f'Benchmarking ({name})...', end='')

		if only_testing or m.trained_weights_exists(suffix):
			m.load_weights(suffix)
		else:
			t0 = time.time()
			m.fit(obj=obj, target_as_input=target_as_input)
			m.save_weights(suffix)
			print(f' done! Time = {(time.time() - t0)/60:.1f} minutes', end='')
		print()

		return m.test(obj=obj, target_as_input=target_as_input)

	def fit_or_load(m, suffix, n_batches, **train_kwargs):

		if only_testing or m.trained_weights_exists(suffix):
			m.load_weights(suffix)

		else:
			t0 = time.time()
			m.train(lr, chunk_size, batch_size, n_batches, **train_kwargs)
			m.save_weights(suffix)
			print(f' done! Time = {(time.time() - t0)/60:.1f} minutes')

	# Observations high bound: memoryless logistic decoder (x_t -> y_t)
	obs_high = logistic_ref('observations high bound', '_observation-high', obj=['obs'])

	# Predictions high bound: global distribution (marginal predictor)
	print('Benchmarking (predictions high bound)...', end='')
	gd = bachbayes.Bachmodel({'mod_type': 'feedforward', 'n_layers': 1, **base_pars})
	pred_high = gd.test_globaldist_model()
	print()
	del gd

	# Predictions Markov high bound: logistic 1-step from noisy input (x_t -> y_{t+1})
	pred_markh = logistic_ref('predictions markov high bound', '_prediction-markov-high', obj=['pred'])

	# Predictions Markov low bound: logistic 1-step from clean input (y_t -> y_{t+1})
	pred_markl = logistic_ref('predictions markov low bound', '_prediction-markov-low',
							  obj=['pred'], target_as_input=True)

	# Predictions low bound: prediction-trained GRU
	pars = {'mod_type': 'gru', 'n_layers': 1}
	pars.update(base_pars)
	m = bachbayes.Bachmodel(pars)

	print('Benchmarking (predictions low bound)...', end='')
	fit_or_load(m, '_prediction-low', max_batches_rnn, obj=['pred'])
	pred_low = m.test(obj=['pred'])
	del m

	# Reporting
	performance  = {'obs_high':   obs_high,
			 		'pred_high':  pred_high,
			 		'pred_markh': pred_markh,
			 		'pred_markl': pred_markl,
			 		'pred_low':   pred_low}

	return performance


def pipeline(only_testing=False, n_runs=N_RUNS_TRAIN):

	save_to_results(noise=NOISE_VALS, n_hidden=N_HIDDEN_VALS, n_runs=N_RUNS)

	for i, noise in enumerate(NOISE_VALS):
		for j, n_hidden in enumerate(N_HIDDEN_VALS):

			print(f'\nModel ({i:02},{j:02}) of ({len(NOISE_VALS)},{len(N_HIDDEN_VALS)}))')
			print('--------------------------------')
			t0 = time.time()

			modname = f'gru_nhidden-{n_hidden:d}_nlayers-1_chromatic_noise{noise}'.replace('.', 'p')

			for run_n in range(n_runs):

				print(f' -- Run {run_n+1}/{n_runs}')

				if not only_testing and os.path.exists(f'./models/{modname}_prediction_run{run_n:02d}.pth'):
					continue

				performance_rnn = main_fitting(noise, n_hidden, run_n=run_n, only_testing=only_testing)
				save_to_results(noise, n_hidden, run_n, **performance_rnn)

			print(f'------ model took {(time.time() - t0)/60:<5.1f}minutes ------')


def baselines_pipeline(only_testing=False):

	save_to_results(noise=NOISE_VALS, n_hidden=N_HIDDEN_VALS, n_runs=N_RUNS)

	for i, noise in enumerate(NOISE_VALS):

		print(f'\nBaselines for noise {noise} ({i+1}/{len(NOISE_VALS)})')
		print('--------------------------------')
		t0 = time.time()

		performance_baseline = baselines(noise, only_testing=only_testing)
		save_to_results(noise, **performance_baseline)

		print(f'------ baselines took {(time.time() - t0)/60:<5.1f}minutes ------')


def save_to_results(this_noise=None, this_n_hidden=None, this_run=None, results_path='./results/results.pickle', **kwargs):

	if not os.path.exists(results_path):
		savedict = {}
	else:
		with open(results_path, 'rb') as f:
			savedict = pickle.load(f)

	# Axis definitions (init call). Refuse to touch a results file that was
	# produced under a different design: it must be archived manually first
	for key in ['noise', 'n_hidden', 'n_runs']:
		if key in kwargs:
			if key in savedict and savedict[key] != kwargs[key]:
				raise ValueError(f'{results_path} has a different "{key}" axis '
								 f'({savedict[key]} vs {kwargs[key]}); archive it first')
			savedict[key] = kwargs[key]

	metrics = {k: v for k, v in kwargs.items() if k not in ['noise', 'n_hidden', 'n_runs']}

	noise, n_hidden, n_runs = savedict['noise'], savedict['n_hidden'], savedict['n_runs']

	for key, value in metrics.items():
		if this_run is not None:
			# Per-run network metric: (noise, n_hidden, run, test work)
			shape = (len(noise), len(n_hidden), n_runs, len(value))
		else:
			# Reference-model metric, one per noise level: (noise, test work)
			shape = (len(noise), len(value))
		if key not in savedict:
			savedict[key] = np.full(shape, np.nan)
		elif savedict[key].shape != shape:
			raise ValueError(f'{results_path} stores "{key}" with shape '
							 f'{savedict[key].shape}, expected {shape}; archive it first')

	if this_noise is not None:
		ix_noise = noise.index(this_noise)
		for key, value in metrics.items():
			if this_run is not None:
				savedict[key][ix_noise, n_hidden.index(this_n_hidden), this_run, :] = value
			else:
				savedict[key][ix_noise, :] = value

	with open(results_path, 'wb') as f:
		pickle.dump(savedict, f)


def prediction_error_analysis_pipeline(n_runs=N_RUNS_TRAIN, pe_path='./results/pe_results.pickle'):

	savedict      = load_results()
	noise_vals    = savedict['noise']
	n_hidden_vals = savedict['n_hidden']
	operas        = sorted(bachbayes.Chunker('./test', 1, None, True, 0, seed=bachbayes.EVAL_SEED).song_pool)

	# Per-composition metrics, all (noise, n_hidden, run, opera): the correlation
	# between innovation magnitude and state-update magnitude, plus the decoding
	# MSE for every target x decoder pair of the ladder
	metric_names = ['pe_std'] + [f'dec_{t}_{d}' for t in PE_TARGETS for d in PE_DECODERS]
	shape        = (len(noise_vals), len(n_hidden_vals), n_runs, len(operas))

	# Resume from an existing pickle if it matches the design; completed cells are
	# skipped, so the sweep can be interrupted and relaunched
	if os.path.exists(pe_path):
		with open(pe_path, 'rb') as f:
			results = pickle.load(f)
		if results.get('noise') != noise_vals or results['pe_std'].shape != shape:
			raise ValueError(f'{pe_path} was produced under a different design; archive it first')
	else:
		results = {name: np.full(shape, np.nan) for name in metric_names}
		results.update({'noise': noise_vals, 'n_hidden': n_hidden_vals, 'n_runs': n_runs})

	for six, noise in enumerate(noise_vals):

		# The corpora and their frozen noise realisations depend on the noise level
		# only, so the chunkers are shared across network sizes and runs
		val_chunker  = bachbayes.Chunker('./validation', 1, None, True, noise, seed=bachbayes.EVAL_SEED)
		test_chunker = bachbayes.Chunker('./test',       1, None, True, noise, seed=bachbayes.EVAL_SEED)

		for nix, n_hidden in enumerate(n_hidden_vals):
			for run_n in range(n_runs):

				if np.isfinite(results['dec_sig_h'][six, nix, run_n]).any():
					continue

				t0 = time.time()
				print(f'Computing PE for (noise={noise:g}, n={n_hidden}, run {run_n})...', end='')

				# A. Load the denoising-trained network with its prediction probe
				pars = {'datapath': '.', 'chromatic': True, 'noise': noise,
						'n_layers': 1, 'n_hidden': n_hidden, 'mod_type': 'gru'}

				m = bachbayes.Bachmodel(pars)
				m.load_weights(f'_prediction_run{run_n:02d}')

				# B. Fit the decoders on the validation split (silence masked). 
				feats = {'x': [], 'h': []}
				targs = {t: [] for t in PE_TARGETS}

				for opera in sorted(val_chunker.song_pool):
					song  = val_chunker.read_song_as_tensor(opera)
					terms = m.compute_pe_terms(song['sample'], song['target'])
					mk    = terms['mask']
					feats['x'].append(terms['x_next'][mk])
					feats['h'].append(terms['stm'][mk])
					for t in PE_TARGETS:
						targs[t].append(terms[f'pe_{t}'][mk])

				feats = {k: np.concatenate(v) for k, v in feats.items()}
				targs = {k: np.concatenate(v) for k, v in targs.items()}

				width    = targs[PE_TARGETS[0]].shape[1]
				stacked  = np.hstack([targs[t] for t in PE_TARGETS])
				reg      = {d: RidgeCV(alphas=np.logspace(-3, 3, 7), alpha_per_target=True)
							   .fit(feats[d], stacked)
							for d in ['x', 'h']}
				cols     = {t: slice(k * width, (k + 1) * width)
							for k, t in enumerate(PE_TARGETS)}
				constant = {t: targs[t].mean(0) for t in PE_TARGETS}

				# C. Evaluate on the test set, one value per composition
				for nop, opera in enumerate(operas):

					song  = test_chunker.read_song_as_tensor(opera)
					terms = m.compute_pe_terms(song['sample'], song['target'])
					mk    = terms['mask']
					if mk.sum() < 10:
						continue

					pe2  = (terms['pe_raw'][mk] ** 2).mean(1)
					std2 = (terms['std'][mk] ** 2).mean(1)
					results['pe_std'][six, nix, run_n, nop] = ss.pearsonr(pe2, std2)[0]

					hat = {d: reg[d].predict(terms['x_next' if d == 'x' else 'stm'][mk])
						   for d in ['x', 'h']}

					for t in PE_TARGETS:
						truth = terms[f'pe_{t}'][mk]
						preds = {'b': constant[t],
								 'x': hat['x'][:, cols[t]],
								 'h': hat['h'][:, cols[t]]}
						for d in PE_DECODERS:
							results[f'dec_{t}_{d}'][six, nix, run_n, nop] = \
								((truth - preds[d]) ** 2).mean()

				# Incremental save: a completed cell survives an interruption
				with open(pe_path, 'wb') as f:
					pickle.dump(results, f)

				print(f' done! Time = {(time.time()-t0)/60:.2f}m')


def load_results(results_path='./results/results.pickle'):

	with open(results_path, 'rb') as f:
		savedict = pickle.load(f)

	return savedict


if __name__ == '__main__':
	# prepare_data()
	pipeline()
	baselines_pipeline()
	# prediction_error_analysis_pipeline()



