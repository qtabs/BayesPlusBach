import os
# Pin to the RTX 4000 Ada (GPU 0). This box also carries a Turing T400 (GPU 1);
# leaving both GPUs visible triggers intermittent cuDNN RNN-backward failures
# (CUDNN_STATUS_INTERNAL_ERROR) on the mixed-architecture setup. Must run before
# torch is imported (via bachbayes) to take effect. Override on the command line
# if ever needed, e.g. CUDA_VISIBLE_DEVICES=1 python3 pipeline.py
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import bachbayes
import glob
import numpy as np
import os
import pickle
import random
import shutil
import time
import scipy.stats as ss
from sklearn.linear_model import LinearRegression


# Sweep design. Noise levels are sampled densely where the single-observation
# discriminability (d' = 1/sigma) crosses unity, sparsely at the asymptotes;
# sigma = 0.001 is the near-noiseless control. Hidden sizes start above the
# stimulus dimensionality (12) so that no cell is a representational bottleneck
NOISE_VALS    = [0.001, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0]
N_HIDDEN_VALS = [8, 16, 32, 64, 128, 256]
N_RUNS        = 10   # storage capacity of the results arrays
N_RUNS_TRAIN  = 5    # runs trained by default (extendable up to N_RUNS for free:
                     # pipeline() resumes per run and figures aggregate with nanmean)
N_REF         = 256  # reference models are capacity-saturated (max of the sweep)
N_ATTEMPTS    = 3    # tries per run before giving up on it and moving on


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

		# Resume at stage granularity. A stage checkpoint is written only once
		# train() has spent its whole batch budget and reloaded the best
		# validation state, so an existing file is always a finished stage and
		# reloading it is exact, not an approximation. Without this, stopping
		# during the 3,000-batch prediction stage discarded the 20,000-batch
		# decoding stage that preceded it.
		# The report suffix is the run alone: _write_report_ already
		# disambiguates the two stages by objective and freeze list
		if m.trained_weights_exists(weight_suffix):
			m.load_weights(weight_suffix)
			print(' already trained; loaded from disk')
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
		print('Training on observations...', end='')
		fit_or_load(f'_observation_run{run_n:02d}', max_batches_obs,
					freeze=['out_pred'], obj=['obs'])

		## Training linear readout on prediction accuracy
		print('Training on predictions...', end='')
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

	# Reference models estimate properties of the task (conditional entropies),
	# so they are computed once per noise level at saturated capacity rather
	# than width-matched to each RNN of the sweep

	# Parameters
	lr = 0.02
	chunk_size = 512
	batch_size = 512
	max_batches_rnn = 25000
	max_batches_ff  = 25000

	base_pars = {'datapath'   : '.',
				 'chunk_size' : chunk_size,
				 'chromatic'  : True,
				 'noise'      : noise,
				 'n_hidden'   : n_hidden}

	def fit_or_load(m, suffix, n_batches, **train_kwargs):
		if only_testing or m.trained_weights_exists(suffix):
			m.load_weights(suffix)
		else:
			t0 = time.time()
			m.train(lr, chunk_size, batch_size, n_batches, **train_kwargs)
			m.save_weights(suffix)
			print(f' done! Time = {(time.time() - t0)/60:.1f} minutes')


	# Observations high-bound -> NN one-step
	pars = {'mod_type': 'feedforward', 'n_layers': 3}
	pars.update(base_pars)
	m = bachbayes.Bachmodel(pars)

	print('Benchmarking (observations high bound)...', end='')
	fit_or_load(m, '_observation-high', max_batches_ff, obj=['obs'])
	obs_high = m.test(obj=['obs'])


	# Predictions high-bound: global distribution
	print('Benchmarking (predictions high bound)...', end='')
	pred_high = m.test_globaldist_model()


	# Predictions markov high bound: 1-back predictions
	pars = {'mod_type': 'feedforward', 'n_layers': 1}
	pars.update(base_pars)
	m = bachbayes.Bachmodel(pars)

	print('\nBenchmarking (predictions markov bound)...', end='')
	fit_or_load(m, '_prediction-markov-high', max_batches_ff, obj=['pred'])
	pred_markh = m.test(obj=['pred'])


	# Predictions markov low bound: 1-back predictions with ground-truth state
	pars = {'mod_type': 'feedforward', 'n_layers': 1}
	pars.update(base_pars)
	m = bachbayes.Bachmodel(pars)

	print('Benchmarking (predictions markov low bound)...', end='')
	fit_or_load(m, '_prediction-markov-low', max_batches_ff, obj=['pred'], target_as_input=True)
	pred_markl = m.test(obj=['pred'], target_as_input=True)


	# Predictions low bound
	pars = {'mod_type': 'gru', 'n_layers': 1}
	pars.update(base_pars)
	m = bachbayes.Bachmodel(pars)

	print('Benchmarking (predictions low bound)...', end='')
	fit_or_load(m, '_prediction-low', max_batches_rnn, obj=['pred'])
	pred_low = m.test(obj=['pred'])


	# ToDo: Use linear regression instead of 1-layer ANNs for the Markov models

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

				for attempt in range(N_ATTEMPTS):
					try:
						performance_rnn = main_fitting(noise, n_hidden, run_n=run_n, only_testing=only_testing)
						save_to_results(noise, n_hidden, run_n, **performance_rnn)
						break
					except RuntimeError as e:
						# Usually memory pressure, which retrying will not clear.
						# After a few tries move on to the next run: this one
						# leaves no checkpoint behind, so re-running pipeline()
						# later picks it up again
						print(f'Attempt {attempt+1}/{N_ATTEMPTS} failed: {e}')

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


def prediction_error_analysis_pipeline(run_n=0):

	operas   = sorted(bachbayes.Chunker('./test', 1, None, True, 0, seed=bachbayes.EVAL_SEED).song_pool)
	savedict = load_results()
	noise_vals    = savedict['noise']
	n_hidden_vals = savedict['n_hidden']

	results = {'pe_stm': np.full((len(noise_vals), len(n_hidden_vals), len(operas)), np.nan),
			   'pe_std': np.full((len(noise_vals), len(n_hidden_vals), len(operas)), np.nan),
			   #'en_m':   np.full((len(noise_vals), len(n_hidden_vals), len(operas), max(n_hidden_vals)), np.nan),
			   #'en_d':   np.full((len(noise_vals), len(n_hidden_vals), len(operas), max(n_hidden_vals)), np.nan),
			   'dec_m':  np.full((len(noise_vals), len(n_hidden_vals), len(operas)), np.nan),
			   'dec_d':  np.full((len(noise_vals), len(n_hidden_vals), len(operas)), np.nan),
			   'dec_b':  np.full((len(noise_vals), len(n_hidden_vals), len(operas)), np.nan),
			   'noise': noise_vals, 'n_hidden': n_hidden_vals}

	for six, noise in enumerate(noise_vals):
		for nix, n_hidden in enumerate(n_hidden_vals):

			t0 = time.time()
			print(f'Computing PE for (noise={noise:.1f}, n={n_hidden})...', end='')


			# A. Load model
			pars = {'datapath'   : '.',
					'chromatic'  : True,
					'noise'      : noise,
					'n_layers'   : 1,
					'n_hidden'   : n_hidden,
					'mod_type'   : 'gru'}

			m = bachbayes.Bachmodel(pars)
			m.load_weights(f'_prediction_run{run_n:02d}')


			# B. Train stats to pe linear regression models
			tr_chunker = bachbayes.Chunker(m.train_path, 1, None, m.chromatic, m.noise, m.dev, seed=bachbayes.EVAL_SEED)

			pe_train, stm_train, std_train = [], [], [] 

			for opera in tr_chunker.song_pool:

				sample = tr_chunker.read_song_as_tensor(opera)['sample']
				pe, stm, std = m.compute_pe_and_state(sample)

				pe_train.append(pe)
				stm_train.append(stm)
				std_train.append(std)

			pe_train  = np.concatenate(pe_train, axis=0)
			stm_train = np.concatenate(stm_train, axis=0)
			std_train = np.concatenate(std_train, axis=0)

			reg_m = LinearRegression().fit(stm_train, pe_train)
			reg_d = LinearRegression().fit(std_train, pe_train)


			# C. Compute PE and stats in the test set
			chunker = bachbayes.Chunker(m.test_path, 1, None, m.chromatic, m.noise, m.dev, seed=bachbayes.EVAL_SEED)
			operas  = sorted(chunker.song_pool)

			for nop, opera in enumerate(operas):

				# 0. Computing prediction error and network's states
				sample = chunker.read_song_as_tensor(opera)['sample']
				pe, stm, std = m.compute_pe_and_state(sample)


				# 1. Total prediction error with network activity
				pe2  = (pe ** 2).mean(1)
				stm2 = (stm ** 2).mean(1) 
				std2 = (std ** 2).mean(1) 

				results['pe_stm'][six, nix, nop] = ss.pearsonr(pe2, stm2)[0]
				results['pe_std'][six, nix, nop] = ss.pearsonr(pe2, std2)[0]

				"""
				# 2. Prediction error encoding per neuron
				for n in range(n_hidden):
					stm2 = stm[..., n] ** 2
					std2 = std[..., n] ** 2
					results['en_m'][six, nix, nop, n] = ss.pearsonr(pe2, stm2)[0]
					results['en_d'][six, nix, nop, n] = ss.pearsonr(pe2, std2)[0]
				"""

				# 3. Prediction error decoding accuracy
				pe_hat_m = reg_m.predict(stm)
				pe_hat_d = reg_d.predict(std)
				pe_hat_b = pe_train.mean(0)

				results['dec_m'][six, nix, nop] = ((pe - pe_hat_m)**2).mean()
				results['dec_d'][six, nix, nop] = ((pe - pe_hat_d)**2).mean()
				results['dec_b'][six, nix, nop] = ((pe - pe_hat_b)**2).mean()


			with open('./results/pe_results.pickle', 'wb') as f:
				pickle.dump(results, f)

			print(f' done! Time = {(time.time()-t0)/60:.2f}m')


	# ToDo: Enconding of PE+/- --> (np.maximum(-pe,  0)**2).mean(1)


def load_results(results_path='./results/results.pickle'):

	with open(results_path, 'rb') as f:
		savedict = pickle.load(f)

	return savedict


if __name__ == '__main__':
	# prepare_data()
	baselines_pipeline()
	pipeline()
	# prediction_error_analysis_pipeline()


# ToDo: 
#  - rename baseline variables to more informative names (e.g. globaldist)


