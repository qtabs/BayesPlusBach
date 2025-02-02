import bachbayes
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import seaborn as sns
import time


def main_fitting(noise, n_hidden):

	pars = {'datapath'   : '.',
			'chunk_size' : 512,
			'chromatic'  : True,
			'noise'      : noise,
			'n_layers'   : 1,
			'n_hidden'   : n_hidden,
			'mod_type'   : 'gru'}

	m = bachbayes.Bachmodel(pars)

	# Training
	lr = 0.02
	chunk_size = 512
	batch_size = 512
	max_batches_obs = 1000
	n_batches_pred  = 500
	n_test_samples  = 4

	## Training RNN on observation accuracy
	print('Training on observations...', end='')
	t0 = time.time()
	m.train(lr, chunk_size, batch_size, max_batches_obs, freeze=['out_pred'], obj=['obs'])
	m.save_weights('_observation')
	print(f' done! Time = {(time.time() - t0)/60:.1f} minutes')

	## Training linear readout on prediction accuracy
	print('Training on predictions...', end='')
	t0 = time.time()
	m.train(lr, chunk_size, batch_size, n_batches_pred, freeze=['rnn', 'out_obs'], obj=['pred'])
	m.save_weights('_prediction')
	print(f' done! Time = {(time.time() - t0)/60:.1f} minutes')

	# Testing
	obs_error  = m.test(chunk_size, batch_size, n_test_samples, obj=['obs'])
	pred_error = m.test(chunk_size, batch_size, n_test_samples, obj=['pred'])

	obs_err_m,  obs_err_e  = obs_error.mean(),  obs_error.std()  #/ n_test_samples**.5
	pred_err_m, pred_err_e = pred_error.mean(), pred_error.std() #/ n_test_samples**.5

	print('\n---------------------------------')
	print(f'| Observation error = {obs_err_m:.2f}' + u'\u00B1' + f'{obs_err_e:.2f} |')
	print(f'| Prediction error  = {pred_err_m:.2f}' + u'\u00B1' + f'{pred_err_e:.2f} |')
	print('---------------------------------\n')

	performance = {'obs_err_m':  obs_err_m,
				   'obs_err_e':  obs_err_e,
				   'pred_err_m': pred_err_m,
				   'pred_err_e': pred_err_e}

	return performance


def benchmarking(noise, n_hidden):

	# Global parameters
	lr = 0.02
	chunk_size = 512
	batch_size = 512
	max_batches_rnn = 5000
	max_batches_ff  = 5000
	n_test_samples  = 4

	base_pars = {'datapath'   : '.',
				 'chunk_size' : chunk_size,
				 'chromatic'  : True,
				 'noise'      : noise,
				 'n_hidden'   : n_hidden}

	# Observations high-bound -> NN one-step
	print('Benchmarking (observations high bound)...', end='')
	t0 = time.time()
	pars = {'mod_type': 'feedforward', 'n_layers': 3}
	pars.update(base_pars)
	m = bachbayes.Bachmodel(pars)
	if not m.trained_weights_exists('_observation-high'):
		m.train(lr, chunk_size, batch_size, max_batches_ff, obj=['obs'])
		m.save_weights('_observation-high')
	obs_high = m.test(chunk_size, batch_size, n_test_samples, obj=['obs'])
	print(f' done! Time = {(time.time() - t0)/60:.1f} minutes')

	# Predictions high-bound: global distribution
	print('Benchmarking (predictions high bound)...', end='')
	t0 = time.time()
	n_train = max(max_batches_rnn, max_batches_ff) * batch_size
	n_test  = n_test_samples * batch_size
	pred_high = bachbayes.test_globaldist_model(base_pars, n_train, n_test)
	print(f' done! Time = {(time.time() - t0)/60:.1f} minutes')

	# Predictions markov high bound: 1-back predictions
	print('Benchmarking (predictions markov bound)...', end='')
	t0 = time.time()
	pars = {'mod_type': 'feedforward', 'n_layers': 3}
	pars.update(base_pars)
	m = bachbayes.Bachmodel(pars)
	if not m.trained_weights_exists('_prediction-markov-high'):
		m.train(lr, chunk_size, batch_size, max_batches_ff, obj=['pred'])
		m.save_weights('_prediction-markov-high')
	pred_markh = m.test(chunk_size, batch_size, n_test_samples, obj=['pred'])
	print(f' done! Time = {(time.time() - t0)/60:.1f} minutes')

	# Predictions markov low bound: 1-back predictions with ground-truth state
	print('Benchmarking (predictions markov low bound)...', end='')
	t0 = time.time()
	pars = {'mod_type': 'feedforward', 'n_layers': 3}
	pars.update(base_pars)
	m = bachbayes.Bachmodel(pars)
	if not m.trained_weights_exists('_prediction-markov-low'):
		m.train(lr, chunk_size, batch_size, max_batches_ff, obj=['pred'], target_as_input=True)
		m.save_weights('_prediction-markov-low')
	pred_markl = m.test(chunk_size, batch_size, n_test_samples, obj=['pred'])
	print(f' done! Time = {(time.time() - t0)/60:.1f} minutes')

	# Predictions low bound 
	print('Benchmarking (predictions low bound)...', end='')
	t0 = time.time()
	pars = {'mod_type': 'gru', 'n_layers': 1}
	pars.update(base_pars)
	m = bachbayes.Bachmodel(pars)
	if not m.trained_weights_exists('_prediction-low'):
		m.train(lr, chunk_size, batch_size, max_batches_rnn, obj=['pred'])
		m.save_weights('_prediction-low')
	pred_low = m.test(chunk_size, batch_size, n_test_samples, obj=['pred'])
	print(f' done! Time = {(time.time() - t0)/60:.1f} minutes')

	obs_high_m,  obs_high_e    = obs_high.mean(),   obs_high.std()
	pred_high_m, pred_high_e   = pred_high.mean(),  pred_high.std()
	pred_markh_m, pred_markh_e = pred_markh.mean(), pred_markh.std()
	pred_markl_m, pred_markl_e = pred_markl.mean(), pred_markl.std()
	pred_low_m,  pred_low_e    = pred_low.mean(),   pred_low.std()

	print('\n----------------------------------------')
	print(f'| Observation high-bound   = {obs_high_m:.2f}'   + u'\u00B1' + f'{obs_high_e:.2f} |')
	print(f'| Predictions high-bound   = {pred_high_m:.2f}'  + u'\u00B1' + f'{pred_high_m:.2f} |')
	print(f'| Predictions markov-bound = {pred_markl_m:.2f}' + u'\u00B1' + f'{pred_markl_e:.2f} |')
	print(f'| Predictions low-bound    = {pred_low_m:.2f}'   + u'\u00B1' + f'{pred_low_e:.2f} |')
	print('----------------------------------------\n')

	benchmarks = {'obs_high_m':   obs_high_m,
				  'obs_high_e':   obs_high_e,
				  'pred_high_m':  pred_high_m,
				  'pred_high_e':  pred_high_e,
				  'pred_markh_m': pred_markh_m,
				  'pred_markh_e': pred_markh_e,
				  'pred_markl_m': pred_markl_m,
				  'pred_markl_e': pred_markl_e,
				  'pred_low_m':   pred_low_m,
				  'pred_low_e':   pred_low_e}

	return benchmarks


def pipeline():

	noise_vals    = [0.001] + [round(0.1 * n, 2) for n in range(1, 21)]
	n_hidden_vals = [2**n for n in range(1, 9)]

	save_to_results(noise=noise_vals, n_hidden=n_hidden_vals)

	for i, noise in enumerate(noise_vals):
		for j, n_hidden in enumerate(n_hidden_vals):
			
			print(f'\nTraining model ({i:02},{j:02}) of ({len(noise_vals)},{len(n_hidden_vals)}))')
			print('--------------------------------')
			t0 = time.time()
			
			savedict = load_results()

			if 'obs_m' not in savedict or np.isnan(savedict['obs_m'][i, j]):
				performance = main_fitting(noise, n_hidden)
				save_to_results(noise, n_hidden, **performance)

			if 'pred_high_m' not in savedict or np.isnan(savedict['pred_high_m'][i, j]):
				benchmarks = benchmarking(noise, n_hidden)
				save_to_results(noise, n_hidden, **benchmarks)

			print(f'------ model took {(time.time() - t0)/60:<5.1f}minutes ------')

def save_to_results(this_noise=None, this_n_hidden=None, **kwargs):

	if not os.path.exists('./models/results.pickle'):
		savedict = {}
	else:
		with open('./models/results.pickle', 'rb') as f:
			savedict = pickle.load(f)

	savedict.update({key: kwargs[key] for key in ['noise', 'n_hidden'] if key in kwargs})

	noise, n_hidden = savedict['noise'], savedict['n_hidden']
	for key in [k for k in kwargs if k not in savedict]: 
		savedict[key] = np.nan * np.ones((len(noise), len(n_hidden)))

	if this_noise is not None and this_n_hidden is not None:
		ix_noise, ix_n = noise.index(this_noise), n_hidden.index(this_n_hidden)
		for key in [k for k in kwargs if k not in ['noise', 'n_hidden']]:
			savedict[key][ix_noise, ix_n] = kwargs[key]

	with open('./models/results.pickle', 'wb') as f:
		pickle.dump(savedict, f)


def load_results():

	with open('./models/results.pickle', 'rb') as f:
		savedict = pickle.load(f)

	return savedict


def plot_errors():

	savedict = load_results()

	noise, n_hidden = np.round(savedict['noise'], 2), savedict['n_hidden']
	obs_m, obs_e, pred_m, pred_e = savedict['obs_m'], savedict['obs_e'], savedict['pred_m'], savedict['pred_e']
	obs_high_m, obs_high_e   = savedict['obs_high_m'],  savedict['obs_high_e']
	pred_high_m, pred_high_e = savedict['pred_high_m'], savedict['pred_high_e']
	pred_mark_m, pred_mark_e = savedict['pred_mark_m'], savedict['pred_mark_e']
	pred_low_m, pred_low_e   = savedict['pred_low_m'],  savedict['pred_low_e']

	obs_n        = obs_m  - obs_high_m   / (.5*obs_e**2  + .5*obs_high_e**2 )**.5
	pred_n       = pred_m - pred_high_m  / (.5*pred_e**2 + .5*pred_high_e**2)**.5
	markov_imp_n = pred_m - pred_mark_m  / (.5*pred_e**2 + .5*pred_mark_e**2)**.5

	d = (pred_m - obs_m) / (.5*obs_e**2 + .5*pred_e**2)**.5
	D = (pred_m - obs_m)

	fig, axs = plt.subplots(3,1)

	sns.heatmap(obs_n.T,  square=True, xticklabels=noise, yticklabels=n_hidden, 
				annot=True, fmt='.2f', vmin=-10, vmax=0, ax=axs[0])
	axs[0].set_ylabel('relative noise')
	axs[0].set_ylabel('hidden units')
	axs[0].set_title('d(observation error - high bound)')

	sns.heatmap(pred_n.T, square=True, xticklabels=noise, yticklabels=n_hidden, 
				annot=True, fmt='.2f', vmin=-0.5, vmax=0, ax=axs[1])
	axs[1].set_ylabel('relative noise')
	axs[1].set_ylabel('hidden units')
	axs[1].set_title('d(prediction error - high bound)')

	sns.heatmap(markov_imp_n.T, square=True, xticklabels=noise, yticklabels=n_hidden, 
				annot=True, fmt='.2f', vmin=-10, vmax=0, ax=axs[2])
	axs[2].set_ylabel('relative noise')
	axs[2].set_ylabel('hidden units')
	axs[2].set_title('d(prediction error - markovian bound)')

	fig.subplots_adjust(left=0.07, bottom=0.05, right=0.99, top=0.95, hspace=0.2)
	fig.set_size_inches(12, 16)
	fig.savefig('errors.png')


pipeline()
plot_errors()




