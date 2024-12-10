import bachbayes
import numpy as np
import pickle

def run_pipeline(noise, n_hidden):

	pars = {'datapath'   : '.',
			'chunk_size' : 512,
			'chromatic'  : True,
			'noise'      : noise,
			'n_layers'   : 1,
			'n_hidden'   : n_hidden,
			'unit_type'  : 'gru'}

	m = bachbayes.Bachmodel(pars)

	# Training
	lr = 0.02
	chunk_size = 512
	batch_size = 512
	max_batches_obs = 5000
	n_batches_pred  = 100
	n_test_samples  = 4

	## Training RNN on observation accuracy
	m.train(lr, chunk_size, batch_size, max_batches_obs, freeze=['out_pred'], obj=['obs'])
	m.save_weights('_observation')

	## Training linear readout on prediction accuracy
	m.train(lr, chunk_size, batch_size, n_batches_pred, freeze=['rnn', 'out_obs'], obj=['pred'])
	m.save_weights('_prediction')

	# Testing
	obs_error = m.test(chunk_size, batch_size, n_test_samples, obj=['obs'])
	pred_error = m.test(chunk_size, batch_size, n_test_samples, obj=['pred'])

	obs_err_m,  obs_err_e  = obs_error.mean(),  obs_error.std()  #/ n_test_samples**.5
	pred_err_m, pred_err_e = pred_error.mean(), pred_error.std() #/ n_test_samples**.5

	print(f'Observation error = {obs_err_m:.2f}' + u'\u00B1' + f'{obs_err_e:.2f}')
	print(f'Prediction error = {pred_err_m:.2f}' + u'\u00B1' + f'{pred_err_e:.2f}')

	return(obs_err_m, obs_err_e, pred_err_m, pred_err_e)


noise_vals    = [0.01] + np.arange(0.1, 2.1, 0.1)
n_hidden_vals = [2**n for n in range(1, 9)]

obs_m,obs_e,pred_m,pred_e = [np.nan*np.ones((len(noise_vals),len(n_hidden_vals))) for _ in range(4)]

for i, noise in enumerate(noise_vals):
	for j, n_hidden in enumerate(n_hidden_vals):
		print(f'\n#######################################')
		print(f'## Training model ({i:02},{j:02}) of ({len(noise_vals)},{len(n_hidden_vals)})) ##')
		print(f'#######################################\n')
		obs_m[i,j], obs_e[i,j], pred_m[i,j], pred_e[i,j] = run_pipeline(noise, n_hidden)

savedict = {'obs_m':    obs_m,
			'obs_e':    obs_e,
			'pred_m':   pred_m,
			'pred_e':   pred_e,
			'noise':    noise,
			'n_hidden': n_hidden}

with open('./models/results.picke', 'wb') as f:
	pickle.dump(savedict, f)

