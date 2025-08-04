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

	operas = [os.path.split(fpath)[1] for fpath in glob.glob(save_dir + '/bwv*.csv')]

	n_test_samples  = int(np.round(len(operas) * test_rat))
	n_val_samples   = int(np.round(len(operas) * validation_rat))

	random.shuffle(operas)

	samples = {'test': operas[:n_test_samples],
			   'validation': operas[n_test_samples:(n_val_samples+n_test_samples)],
			   'training': operas[(n_val_samples+n_test_samples):]}

	for key in samples:
		sample_dir = os.path.join(basedir, key)
		os.makedirs(sample_dir, exist_ok=True)
		for opera in samples[key]:
			shutil.copy(os.path.join(save_dir, opera), os.path.join(sample_dir, opera))


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


	# Training
	if only_testing:
		m.load_weights(f'_prediction_run{run_n:02d}')

	else:
		## Training RNN on observation accuracy
		print('Training on observations...', end='')
		t0 = time.time()
		m.train(lr, chunk_size, batch_size, max_batches_obs, freeze=['out_pred'], obj=['obs'])
		m.save_weights(f'_observation_run{run_n:02d}')
		print(f' done! Time = {(time.time() - t0)/60:.1f} minutes')

		## Training linear readout on prediction accuracy
		print('Training on predictions...', end='')
		t0 = time.time()
		m.train(lr, chunk_size, batch_size, n_batches_pred, freeze=['rnn', 'out_obs'], obj=['pred'])
		m.save_weights(f'_prediction_run{run_n:02d}')
		print(f' done! Time = {(time.time() - t0)/60:.1f} minutes')


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


def baselines(noise, n_hidden, only_testing=False):

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


	# Observations high-bound -> NN one-step
	pars = {'mod_type': 'feedforward', 'n_layers': 3}
	pars.update(base_pars)
	m = bachbayes.Bachmodel(pars)

	if only_testing:
		m.load_weights('_observation-high')
	else:
		print('Benchmarking (observations high bound)...', end='')
		t0 = time.time()
		if not m.trained_weights_exists('_observation-high'):
			m.train(lr, chunk_size, batch_size, max_batches_ff, obj=['obs'])
			m.save_weights('_observation-high')
		print(f' done! Time = {(time.time() - t0)/60:.1f} minutes')
	
	obs_high = m.test(obj=['obs'])


	# Predictions high-bound: global distribution
	print('Benchmarking (predictions high bound)...', end='')
	t0 = time.time()
	pred_high = m.test_globaldist_model()
	print(f' done! Time = {(time.time() - t0)/60:.1f} minutes')


	# Predictions markov high bound: 1-back predictions
	pars = {'mod_type': 'feedforward', 'n_layers': 1}
	pars.update(base_pars)
	m = bachbayes.Bachmodel(pars)
	
	if only_testing:
		m.load_weights('_prediction-markov-high')
	else:
		print('Benchmarking (predictions markov bound)...', end='')
		t0 = time.time()
		m.train(lr, chunk_size, batch_size, max_batches_ff, obj=['pred'])
		m.save_weights('_prediction-markov-high')
		print(f' done! Time = {(time.time() - t0)/60:.1f} minutes')
			
	pred_markh = m.test(obj=['pred'])


	# Predictions markov low bound: 1-back predictions with ground-truth state
	pars = {'mod_type': 'feedforward', 'n_layers': 1}
	pars.update(base_pars)
	m = bachbayes.Bachmodel(pars)

	if only_testing:
		m.load_weights('_prediction-markov-low')
	else:
		print('Benchmarking (predictions markov low bound)...', end='')
		t0 = time.time()
		m.train(lr, chunk_size, batch_size, max_batches_ff, obj=['pred'], target_as_input=True)
		m.save_weights('_prediction-markov-low')
		print(f' done! Time = {(time.time() - t0)/60:.1f} minutes')
		
	pred_markl = m.test(obj=['pred'])


	# Predictions low bound 
	pars = {'mod_type': 'gru', 'n_layers': 1}
	pars.update(base_pars)
	m = bachbayes.Bachmodel(pars)
	
	if only_testing:
		m.load_weights('_prediction-low')
	else:
		print('Benchmarking (predictions low bound)...', end='')
		t0 = time.time()
		m.train(lr, chunk_size, batch_size, max_batches_rnn, obj=['pred'])
		m.save_weights('_prediction-low')
		print(f' done! Time = {(time.time() - t0)/60:.1f} minutes')
	
	pred_low = m.test(obj=['pred'])


	# ToDo: Use linear regression instead of 1-layer ANNs for the Markov models

	# Reporting
	performance  = {'obs_high':   obs_high,
			 		'pred_high':  pred_high,
			 		'pred_markh': pred_markh,
			 		'pred_markl': pred_markl,
			 		'pred_low':   pred_low}


	return performance


def pipeline(only_testing=False):

	noise_vals    = [0.001] + [round(0.1 * n, 2) for n in range(1, 21)]
	n_hidden_vals = [2**n for n in range(1, 9)]
	n_runs = 9

	# save_to_results(noise=noise_vals, n_hidden=n_hidden_vals)

	for i, noise in enumerate(noise_vals):
		for j, n_hidden in enumerate(n_hidden_vals):
			
			print(f'\nModel ({i:02},{j:02}) of ({len(noise_vals)},{len(n_hidden_vals)}))')
			print('--------------------------------')
			t0 = time.time()
			
			modname = f'gru_nhidden-{n_hidden:d}_nlayers-1_chromatic_noise{noise}'.replace('.', 'p')

			for run_n in range(n_runs):

				print(f' -- Run {run_n+1}/{n_runs}')
				#savedict = load_results()

				done = os.path.exists(f'./models/{modname}_prediction_run{run_n:02d}.pth')
				while not done:
					try:
						performance_rnn = main_fitting(noise, n_hidden, only_testing, run_n)
						done = True
					except KeyboardInterrupt:
						raise
					except RuntimeError as e:
						print(f'Non-critical error: {e}; re-running...')
						continue
				
				# save_to_results(noise, n_hidden, run_n, **performance_rnn)

				#if 'pred_high_m' not in savedict or np.isnan(savedict['pred_high_m'][i, j, 0]):
				#performance_baseline = baselines(noise, n_hidden, only_testing)
				#save_to_results(noise, n_hidden, **performance_baseline)


			print(f'------ model took {(time.time() - t0)/60:<5.1f}minutes ------')


def save_to_results(this_noise=None, this_n_hidden=None, **kwargs):

	if not os.path.exists('./results/results.pickle'):
		savedict = {}
	else:
		with open('./results/results.pickle', 'rb') as f:
			savedict = pickle.load(f)

	savedict.update({key: kwargs[key] for key in ['noise', 'n_hidden'] if key in kwargs})

	noise, n_hidden = savedict['noise'], savedict['n_hidden']
	for key in [k for k in kwargs if k not in savedict]: 
		savedict[key] = np.nan * np.ones((len(noise), len(n_hidden), len(kwargs[key])))

	if this_noise is not None and this_n_hidden is not None:
		ix_noise, ix_n = noise.index(this_noise), n_hidden.index(this_n_hidden)
		for key in [k for k in kwargs if k not in ['noise', 'n_hidden']]:
			savedict[key][ix_noise, ix_n, :] = kwargs[key]

	with open(f'./results/results.pickle', 'wb') as f:
		pickle.dump(savedict, f)


def prediction_error_analysis_pipeline():

	operas   = sorted(bachbayes.Chunker('./test', 1, None, True,0).song_pool)
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
			m.load_weights('_prediction')


			# B. Train stats to pe linear regression models
			tr_chunker = bachbayes.Chunker(m.train_path, 1, None, m.chromatic, m.noise, m.dev)

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
			chunker = bachbayes.Chunker(m.test_path, 1, None, m.chromatic, m.noise, m.dev)
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


def load_results():

	with open('./results/results.pickle', 'rb') as f:
		savedict = pickle.load(f)

	return savedict


if __name__ == '__main__':
	# prepare_data()
	pipeline()
	# pipeline(only_testing=True)
	# prediction_error_analysis_pipeline()


# ToDo: 
#  - rename baseline variables to more informative names (e.g. globaldist)


