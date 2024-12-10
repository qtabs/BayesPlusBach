
import bachbayes

# A. Model definition

pars = {'datapath'   : '.',
		'chunk_size' : 512,
		'chromatic'  : True,
		'noise'      : 1,
		'n_layers'   : 1,
		'n_hidden'   : 32,
		'unit_type'  : 'gru'}

m = bachbayes.Bachmodel(pars)


# B. Training

lr = 0.02
chunk_size = 512
batch_size = 512
n_batches_obs  = 500
n_batches_pred = 100
n_test_samples = 4

## B1. Training RNN on observation accuracy
m.train(lr, chunk_size, batch_size, n_batches_obs, freeze=['out_pred'], obj=['obs'])
m.save_weights('_observation')

## B2. Training linear readout on prediction accuracy
m.train(lr, chunk_size, batch_size, n_batches_pred, freeze=['rnn', 'out_obs'], obj=['pred'])
m.save_weights('_prediction')


# C. Testing

obs_error = m.test(chunk_size, batch_size, n_test_samples, obj=['obs'])
pred_error = m.test(chunk_size, batch_size, n_test_samples, obj=['pred'])

obs_err_m,  obs_err_e  = obs_error.mean(),  obs_error.std()  / n_test_samples**.5
pred_err_m, pred_err_e = pred_error.mean(), pred_error.std() / n_test_samples**.5

print(f'Observation error = {obs_err_m:.2f}' + u'\u00B1' + f'{obs_err_e:.2f}')
print(f'Prediction error = {pred_err_m:.2f}' + u'\u00B1' + f'{pred_err_e:.2f}')