import bachbayes
import pickle
import pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def plot_performance_example(noise, n_hidden):
	
	savedict = pipeline.load_results()
	
	i, j   = savedict['noise'].index(noise), savedict['n_hidden'].index(n_hidden)
	exdict = {k: savedict[k][i, j] for k in savedict if k not in ['noise', 'n_hidden']}
	df     = pd.DataFrame.from_dict(exdict)

	fig, axs = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [2/3, 1]})

	# Observations
	sns.violinplot(df[['obs_high', 'obs_err']], ax=axs[0])
	axs[0].set_xticks(range(2))
	axs[0].set_xticklabels(['Memory-less ANN', 'RNN'])
	axs[0].set_ylabel('Binary Cross Entropy (error)')
	axs[0].set_title('Observations')

	_plot_point_lines_(axs[0], df, 'obs_high', 'obs_err', x0=0, x1=1)
	_write_stats_pair_(axs[0], df,  'obs_high', 'obs_err', x0=0, x1=1, y=0.55)

	# Predictions
	sns.violinplot(df[['pred_markh', 'pred_err', 'pred_high']], ax=axs[1])
	axs[1].set_xticks(range(3))
	axs[1].set_xticklabels(['1st order markov', 'RNN', 'global distribution'])
	axs[1].set_ylabel('Binary Cross Entropy (error)')
	axs[1].set_title('Predictions')

	_plot_point_lines_(axs[1], df, 'pred_err', 'pred_markh', x0=1, x1=0)
	_plot_point_lines_(axs[1], df, 'pred_err', 'pred_high', x0=1, x1=2)
	_write_stats_pair_(axs[1], df, 'pred_err', 'pred_markh', x0=1, x1=0, y=0.65)
	_write_stats_pair_(axs[1], df, 'pred_err', 'pred_high',  x0=1, x1=2, y=0.19, up=False)

	# Saving
	fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.93, hspace=0.2)
	fig.set_size_inches(10, 4)
	modelname = 'noise-' + f'{noise:.1f}'.replace('.', 'p') + f'_n-{n_hidden}'
	fig.savefig(f'./results/performance_example_{modelname}.pdf')


def plot_perfomance_summary(main, baseline, vmax=1, plot_cohen=True):

	savedict = pipeline.load_results()

	values = np.empty((len(savedict['n_hidden']), len(savedict['noise'])))
	annot  = np.empty((len(savedict['n_hidden']), len(savedict['noise']))).astype(str)
	bonferroni = len(savedict['noise']) * len(savedict['n_hidden'])

	for six, noise in enumerate(savedict['noise']):
		for nix, n_hidden in enumerate(savedict['n_hidden']):
			
			x, y = savedict[main][six, nix], savedict[baseline][six, nix]
			cohen, pval = _compute_stats_(x, y)
			values[nix, six] = cohen if plot_cohen else (x-y).mean()
			annot[nix, six]  = _beautify_annot_(values[nix, six], pval * bonferroni)

	fig, ax = plt.subplots(1, 1)
	sns.heatmap(values, cmap='RdBu', center=0, vmin=-vmax, vmax=vmax,  ax=ax,
		 		annot=annot, fmt="", annot_kws={'fontsize': 6}, linewidths=0.1, 
		 		cbar=True, cbar_kws={"pad": 0.02}, square=False, 
		 		xticklabels=savedict['noise'], yticklabels=savedict['n_hidden'])
	ax.set_ylabel('n hidden units')
	ax.set_ylabel('noise amplitude')

	fig.subplots_adjust(left=0.03, bottom=0.1, right=1.13, top=0.95)
	fig.set_size_inches(20, 5)
	fig.savefig(f'./results/summary_perf_{main}-{baseline}-{"cohens" if plot_cohen else "nominal"}.pdf')
	plt.close()


def plot_prederr_example_opera(noise, n_hidden, ex_opera=14):

	# Load model and chunker
	pars = {'datapath'   : '.',
			'chromatic'  : True,
			'noise'      : noise,
			'n_layers'   : 1,
			'n_hidden'   : n_hidden,
			'mod_type'   : 'gru'}

	m = bachbayes.Bachmodel(pars)
	m.load_weights('_prediction')
	
	chunker = bachbayes.Chunker(m.test_path, 1, None, m.chromatic, m.noise, m.dev)
	operas = sorted(chunker.song_pool)
	opname = operas[ex_opera].upper()

	# Time series of the states and prediction error for ex_opera
	sample = chunker.read_song_as_tensor(operas[ex_opera])['sample']
	pe, stm, std = m.compute_pe_and_state(sample)
	
	pe2, stm2, std2 = (pe**2).mean(1), (stm**2).mean(1), (std**2).mean(1)

	
	# Correlations
	rho_m, pval_m = ss.pearsonr(pe2, stm2)
	rho_d, pval_d = ss.pearsonr(pe2, std2)


	# Plotting
	fig, axs = plt.subplots(1, 3, gridspec_kw={'width_ratios': [3, 1, 1]})
	axs[0].plot((pe2 - pe2.mean()) / pe2.std(),    color='tab:orange', label='prediction error L2')
	axs[0].plot((stm2 - stm2.mean()) / stm2.std(), color='tab:blue', label='hidden states L2')
	axs[0].plot((std2 - std2.mean()) / std2.std(), color='tab:green', label='derivative of hidden states L2')
	axs[0].legend()
	axs[0].set_title(f'{opname} (testing set) - timecourses')
	axs[0].set_xlabel('token number')
	axs[0].set_ylabel('z-score')
	axs[0].set_xlim([0, len(pe2)-1])

	sns.regplot(x=stm2, y=pe2, ax=axs[1], color='tab:blue')
	axs[1].set_ylabel('prediction error L2')
	axs[1].set_xlabel('hidden states L2')
	axs[1].set_title(f'States: correlation r={rho_m:.2f} (p={pval_m:.2g})')

	sns.regplot(x=std2, y=pe2, ax=axs[2], color='tab:green')
	axs[2].set_ylabel('prediction error L2')
	axs[2].set_xlabel('derivative of hidden states L2')
	axs[2].set_title(f'Derivatives: correlation r={rho_d:.2f} (p={pval_d:.2g})')
	

	# Saving
	fig.subplots_adjust(left=0.03, right=0.99, bottom=0.15, top=0.93, wspace=0.16)
	fig.set_size_inches(20, 4)
	modelname = 'noise-' + f'{noise:.1f}'.replace('.', 'p') + f'_n-{n_hidden}'
	fig.savefig(f'./results/pederr_example_{modelname}_{opname}.pdf')


def plot_prederr_example(noise, n_hidden):

	with open('./results/pe_results.pickle', 'rb') as f:
		pe_dict = pickle.load(f)

	i, j   = pe_dict['noise'].index(noise), pe_dict['n_hidden'].index(n_hidden)
	exdict = {k: pe_dict[k][i, j] for k in pe_dict if k not in ['noise', 'n_hidden']}
	df     = pd.DataFrame.from_dict(exdict)

	fig, axs = plt.subplots(1, 2, sharey=False, gridspec_kw={'width_ratios': [2/3, 1]})

	# Correlation
	sns.violinplot(df[['pe_stm', 'pe_std']], ax=axs[0])
	axs[0].set_xticks(range(2))
	axs[0].set_xticklabels(['States', 'Derivatives'])
	axs[0].set_ylabel('Correlation coefficient')
	axs[0].set_title('Correlation between PE and activity')
	axs[0].set_ylim([0, 0.75])

	_write_stats_point_(axs[0], df, 'pe_stm', x=0, y=0.55)
	_write_stats_point_(axs[0], df, 'pe_std', x=1, y=0.72)


	# Decoding error
	sns.violinplot(df[['dec_m', 'dec_b', 'dec_d']], ax=axs[1])
	axs[1].set_xticks(range(3))
	axs[1].set_xticklabels(['States', 'Baseline', 'Derivatives'])
	axs[1].set_ylabel('PE decoding error')
	axs[1].set_title('PE encoding')
	axs[1].set_ylim([0.04, 0.78])

	_plot_point_lines_(axs[1], df, 'dec_b', 'dec_m', x0=1, x1=0)
	_plot_point_lines_(axs[1], df, 'dec_b', 'dec_d', x0=1, x1=2)
	_write_stats_pair_(axs[1], df, 'dec_b', 'dec_m', x0=1, x1=0, y=0.71)
	_write_stats_pair_(axs[1], df, 'dec_b', 'dec_d', x0=1, x1=2, y=0.12, up=False)

 
	# Saving
	fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.93, hspace=0.2)
	fig.set_size_inches(10, 4)
	modelname = 'noise-' + f'{noise:.1f}'.replace('.', 'p') + f'_n-{n_hidden}'
	fig.savefig(f'./results/prederr_example_{modelname}.pdf')


def plot_prederr_summary(main, baseline=None, vmax=1, plot_cohen=True):

	with open('./results/pe_results.pickle', 'rb') as f:
		pe_dict = pickle.load(f)

	savedict = pipeline.load_results()

	values = np.empty((len(pe_dict['n_hidden']), len(pe_dict['noise'])))
	annot  = np.empty((len(pe_dict['n_hidden']), len(pe_dict['noise']))).astype(str)
	bonferroni = len(pe_dict['noise']) * len(pe_dict['n_hidden'])

	for six, noise in enumerate(pe_dict['noise']):
		for nix, n_hidden in enumerate(pe_dict['n_hidden']):
			
			x = pe_dict[main][six, nix]
			y = pe_dict[baseline][six, nix] if baseline is not None else None
			cohen, pval = _compute_stats_(x, y)
			
			if y is None:
				values[nix, six] = cohen if plot_cohen else (x).mean()
			else:
				values[nix, six] = cohen if plot_cohen else (x-y).mean()

			annot[nix, six]  = _beautify_annot_(values[nix, six], pval * bonferroni)

	fig, ax = plt.subplots(1, 1)
	sns.heatmap(values, cmap='RdBu', center=0, vmin=-vmax, vmax=vmax,  ax=ax,
		 		annot=annot, fmt="", annot_kws={'fontsize': 6}, linewidths=0.1, 
		 		cbar=True, cbar_kws={"pad": 0.02}, square=False, 
		 		xticklabels=pe_dict['noise'], yticklabels=pe_dict['n_hidden'])
	ax.set_ylabel('n hidden units')
	ax.set_ylabel('noise amplitude')

	fig.subplots_adjust(left=0.03, bottom=0.12, right=1.05, top=0.95)
	fig.set_size_inches(20, 5)
	fig.savefig(f'./results/summary_prederr_{main}-{baseline}-{"cohens" if plot_cohen else "nominal"}.pdf')
	plt.close()


def _write_stats_pair_(ax, df, col0, col1, x0, x1, y, up=True):

	cohens_d, p_val = _compute_stats_(df[col0], df[col1])

	y0, y1 = y, y + (1 if up else -1) * 0.02
	text = f'd = {cohens_d:.2f} (p = {p_val:.1g})'

	ax.plot([x0, x0], [y0, y1], 'k-', linewidth=0.7)
	ax.plot([x0, x1], [y1, y1], 'k-', linewidth=0.7)
	ax.plot([x1, x1], [y1, y0], 'k-', linewidth=0.7)

	sign_options = {'fontsize': 7, 
					'horizontalalignment': 'center',
					'verticalalignment': 'center'}

	y_text = y + (1 if up else -1) * 0.04
	x_text = (x0 + x1) / 2
	ax.text(x_text, y_text, text, **sign_options)


def _write_stats_point_(ax, df, col, x, y, up=True):

	cohens_d, p_val = _compute_stats_(df[col])

	text = f'd = {cohens_d:.2f} (p = {p_val:.1g})'

	sign_options = {'fontsize': 7, 
					'horizontalalignment': 'center',
					'verticalalignment': 'center'}

	ax.text(x, y, text, **sign_options)


def _compute_stats_(x, y=None):
	
	p_val = ss.wilcoxon(x, y).pvalue

	if y is None:
		cohens_d = x.mean() / x.std()
	else:
		cohens_d = (x.mean() - y.mean()) / ((x.std()**2 + y.std()**2) / 2)**.5 

	return cohens_d, p_val


def _beautify_annot_(value, pval):

	if pval > 0.05:
		pval_str = 'ns'
	elif pval > 0.001:
		pval_str = f'p={pval:.1f}'
	elif pval > 10E-10:
		pval_str = f'p={pval:.1g}'
	else:
		pval_str = f'p<1e-10'

	if abs(value) < 0.02:
		val_str = f'{value:.2g}'
	elif abs(value) < 0.2:
		val_str = f'{value:.3f}'
	elif abs(value) < 2:
		val_str = f'{value:.2f}'
	elif abs(value) < 20:
		val_str = f'{value:.1f}'
	elif abs(value) <= 100:
		val_str = f'{value:.0f}'
	else:
		val_str = f'<-100' if value < 0 else f'>100'

	annot = f'{val_str} ({pval_str})'
	
	return annot


def _plot_point_lines_(ax, df, col0, col1, x0, x1):

	options = {'color': 'k', 'linewidth': 0.1, 'alpha': 0.2}

	for y0, y1 in zip(df[col0], df[col1]):
		ax.plot([x0, x1], [y0, y1], **options)



if __name__ == '__main__':
	
	plot_performance_example(0.7, 64)
	
	plot_perfomance_summary('obs_high',   'obs_err',  vmax=10,  plot_cohen=True)
	plot_perfomance_summary('obs_high',   'obs_err',  vmax=0.5, plot_cohen=False)
	plot_perfomance_summary('pred_high',  'pred_err', vmax=10,  plot_cohen=True)
	plot_perfomance_summary('pred_high',  'pred_err', vmax=0.5, plot_cohen=False)
	plot_perfomance_summary('pred_markh', 'pred_err', vmax=5,   plot_cohen=True)
	plot_perfomance_summary('pred_markh', 'pred_err', vmax=0.1, plot_cohen=False)

	plot_prederr_example_opera(0.7, 64, 14)
	plot_prederr_example(0.7, 64)

	plot_prederr_summary('pe_stm', vmax=20, plot_cohen=True)
	plot_prederr_summary('pe_std', vmax=20, plot_cohen=True)
	plot_prederr_summary('dec_b', 'dec_m', vmax=20, plot_cohen=True)
	plot_prederr_summary('dec_b', 'dec_d', vmax=20, plot_cohen=True)



