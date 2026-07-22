"""Figure factory for the manuscript.

Every ``figNN_*`` (main) and ``figSN_*`` (supplementary) function in this module
produces one publication-ready PDF under ``results/``. Running the module executes
all of them in name order, so the figure number lives in the function name.

The Results section presents two effects, each shown first in a single network and
then across the noise x network-size sweep:

  Fig 1  task, architecture, and evidence the denoising task was solved
  Fig 2  prediction emerges - single network
  Fig 3  prediction emerges - across the sweep
  Fig 4  prediction errors are encoded - single network
  Fig 5  prediction errors are encoded - across the sweep
  S1     training loss timecourses
  S2     decoding across the sweep
  S3     the prediction effect replicates across the five runs
  S4     PE encoding replicates across the five runs

Most of these cannot be drawn yet: the sweep is one noise level deep, no reference
models have been trained, and the prediction-error analysis has never been run.
Those functions carry their availability guard and panel plan but an empty body;
they announce what they are waiting for instead of failing.

All figures are laid out to PLOS Computational Biology specifications: full column
width (7.5 in), Arial at 8-10 pt, and fonts embedded as TrueType.
"""

import glob
import os
import pickle
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

import pipeline


# PLOS figure specification. Full-width figures are 7.5 in (2250 px at 300 dpi);
# height must not exceed 8.75 in. Text must be Arial/Times/Symbol at 8-12 pt
FIG_WIDTH     = 7.5
FIG_MAXHEIGHT = 8.75
FONT_SIZE     = 8
LABEL_SIZE    = 9
PANEL_SIZE    = 10

MODELS_DIR  = './models'
RESULTS_DIR = './results'
PE_RESULTS  = './results/pe_results.pickle'

# The cell shown by every single-network figure. Kept here rather than passed as
# arguments so that all example panels are guaranteed to describe the same network
EXAMPLE_NOISE   = 0.8
EXAMPLE_NHIDDEN = 64
EXAMPLE_RUN     = 0
EXAMPLE_OPERA   = 14
EXAMPLE_LENGTH  = 120   # time steps shown in the piano-roll panels

# Truncation of the sequential colormap used for noise levels (see _noise_colors_)
CMAP_LIGHT_END = 0.60
CMAP_DARK_END  = 0.00

# Training reports are written with four decimals (see Bachmodel._write_report_),
# so losses below 5e-5 are stored as exactly 0. Keep the y axis linear: a log
# axis would silently drop the whole floor of the low-noise denoising curves
REPORT_RESOLUTION = 1e-4

# Display names for the keys stored in results.pickle. The stored names are
# counter-intuitive - 'pred_high' is the global distribution, i.e. the *worst*
# predictor, and 'pred_low' the prediction-trained GRU, the best - but renaming
# them in the pickle would invalidate every result already computed, so the
# translation lives here instead
MODEL_LABELS = {
	'obs_err'   : 'RNN',
	'obs_high'  : 'memory-less ANN',
	'pred_err'  : 'RNN (probed)',
	'pred_low'  : 'prediction-trained GRU',
	'pred_markl': '1st-order Markov\n(clean input)',
	'pred_markh': '1st-order Markov\n(noisy input)',
	'pred_high' : 'global distribution',
}

# Reference ladder for prediction, ordered by the error each model is expected to
# reach (methods.tex Eq 5). The probed RNN sits in the middle: beating Markov-high
# is the claim, and whether it also beats Markov-low is the open question
PREDICTION_LADDER = ['pred_low', 'pred_markl', 'pred_err', 'pred_markh', 'pred_high']
DECODING_LADDER   = ['obs_err', 'obs_high']


# Main figures
def fig01_task_and_decoding(fname='fig01_task-and-decoding.pdf'):
	"""Task, architecture, and the decoding result.

	A  architecture schematic - GRU with its two heads and the stage-1/stage-2
	   freezing (drawn by hand externally; the function leaves a blank axis)
	B  example stimulus: clean piano roll y_t over the noisy observation x_t
	C  the same excerpt, target versus decoded y-hat_t
	D  decoding error over the 140 test compositions, RNN versus memory-less ANN
	"""
	_set_plos_style_()

	savedict = _load_results_()
	if not _require_(savedict, DECODING_LADDER, 'results.pickle'):
		return

	print('  data present; panels not implemented yet')


def fig02_prediction_one_net(fname='fig02_prediction-one-net.pdf'):
	"""Prediction emerges in a single network.

	A  excerpt of the upcoming chord y_{t+1} against the network's prediction p_t
	B  the reference ladder over the 140 test compositions, with paired lines and
	   stats brackets for RNN-vs-Markov-high and RNN-vs-global-distribution
	C  per-composition scatter of RNN against Markov-high error, with identity
	   line, showing the effect is not carried by a handful of compositions
	"""
	_set_plos_style_()

	savedict = _load_results_()
	if not _require_(savedict, PREDICTION_LADDER, 'results.pickle'):
		return

	print('  data present; panels not implemented yet')


def fig03_prediction_sweep(fname='fig03_prediction-sweep.pdf'):
	"""Prediction across the noise x network-size sweep.

	A  error against noise amplitude at the exemplar network size, one line per
	   ladder model - shows how the gap opens as the channel degrades
	B  heatmap of Cohen's d, RNN versus Markov-high, over the 9 x 6 design
	C  heatmap of the same contrast in raw BCE units
	"""
	_set_plos_style_()

	savedict = _load_results_()
	if not _require_(savedict, PREDICTION_LADDER, 'results.pickle'):
		return

	print('  data present; panels not implemented yet')


def fig04_pe_one_net(fname='fig04_pe-one-net.pdf'):
	"""Prediction errors are encoded, in a single network.

	A  z-scored timecourses of the population prediction error, the state energy
	   and the state-change energy for one test composition
	B  regression of squared prediction error on state energy
	C  the same against state-change energy
	D  prediction-error decoding MSE over the 140 compositions: states versus
	   derivatives versus the mean-vector baseline decoder
	"""
	_set_plos_style_()

	pe_dict = _load_pe_results_()
	if not _require_(pe_dict, ['pe_stm', 'pe_std', 'dec_m', 'dec_d', 'dec_b'],
					 'pe_results.pickle'):
		return

	print('  data present; panels not implemented yet')


def fig05_pe_sweep(fname='fig05_pe-sweep.pdf'):
	"""Prediction-error encoding across the sweep.

	A  heatmap of the prediction-error / state-energy correlation, 9 x 6
	B  heatmap of the prediction-error / state-change correlation
	C  heatmap of Cohen's d for baseline versus state-based PE decoding
	"""
	_set_plos_style_()

	pe_dict = _load_pe_results_()
	if not _require_(pe_dict, ['pe_stm', 'pe_std', 'dec_m', 'dec_b'],
					 'pe_results.pickle'):
		return

	print('  data present; panels not implemented yet')


# Supplementary figures
def figS1_loss_timecourses(fname='figS1_loss-timecourses.pdf'):
	"""Training loss over the full two-stage schedule.

	One panel per network size; noise levels in colour; the five runs of each
	condition superimposed as thin lines. Denoising (recurrent weights plus the
	observation readout) and prediction (the readout alone, on frozen recurrent
	weights) are concatenated along x so each panel reads as one training history.
	"""
	_set_plos_style_()

	histories = _discover_loss_histories_()
	if not histories:
		print('  no training reports found; skipping')
		return

	n_hidden_vals = sorted({k[0] for k in histories})
	noise_vals    = sorted({k[1] for k in histories})
	colors        = _noise_colors_()

	# Common stage boundary so that panels are directly comparable, even if some
	# runs stopped early
	obs_len  = max((len(h['obs'])  for h in histories.values() if 'obs'  in h), default=0)
	pred_len = max((len(h['pred']) for h in histories.values() if 'pred' in h), default=0)

	fig, axs = plt.subplots(2, 3, figsize=(FIG_WIDTH, 4.4), sharex=True, sharey=True)
	axs = axs.ravel()

	if len(n_hidden_vals) > len(axs):
		print(f'  {len(n_hidden_vals)} hidden sizes found but the layout holds '
			  f'{len(axs)}; plotting the smallest {len(axs)}')
		n_hidden_vals = n_hidden_vals[:len(axs)]

	for pix, ax in enumerate(axs):

		if pix >= len(n_hidden_vals):
			ax.axis('off')
			continue

		n_hidden = n_hidden_vals[pix]

		for noise in noise_vals:
			for (nh, ns, run), hist in sorted(histories.items()):
				if nh != n_hidden or ns != noise:
					continue
				# Runs of the same condition share a colour; they are thin and
				# superimposed to show run-to-run spread rather than a mean.
				# zorder clears the spines (2.5) so that the low-noise curves,
				# which sit flat on zero, stay visible over the x axis rather
				# than being painted over by it
				line_opts = {'color': colors(noise), 'linewidth': 0.35,
							 'alpha': 0.85, 'solid_joinstyle': 'round',
							 'zorder': 3, 'clip_on': False}
				if 'obs' in hist:
					x, y = _downsample_(np.arange(1, len(hist['obs']) + 1), hist['obs'])
					ax.plot(x, y, **line_opts)
				if 'pred' in hist:
					x, y = _downsample_(obs_len + np.arange(1, len(hist['pred']) + 1),
										hist['pred'])
					ax.plot(x, y, **line_opts)

		if pred_len:
			ax.axvline(obs_len, color='0.6', linewidth=0.5, zorder=0)

		ax.set_title(f'{n_hidden} hidden units', fontsize=LABEL_SIZE, pad=3)

	# Stage annotation, once, on the first panel. Positioned in axis-fraction
	# height so that it sits inside the panel rather than under the title
	if pred_len:
		stage_opts = {'fontsize': FONT_SIZE - 1, 'color': '0.4', 'va': 'top',
					  'transform': axs[0].get_xaxis_transform()}
		axs[0].text(obs_len * 0.5, 0.99, 'denoising', ha='center', **stage_opts)
		axs[0].text(obs_len + pred_len * 0.5, 0.99, 'prediction', ha='left',
					rotation=90, **stage_opts)

	axs[0].set_xlim([0, obs_len + pred_len])
	axs[0].set_ylim(bottom=0)
	_set_stage_xticks_(axs[0], obs_len, pred_len)

	for ax in axs[3:]:
		ax.set_xlabel('training batch', fontsize=LABEL_SIZE)
	for ax in axs[::3]:
		ax.set_ylabel('loss (BCE)', fontsize=LABEL_SIZE)

	handles = [plt.Line2D([], [], color=colors(n), linewidth=1) for n in noise_vals]
	labels  = [f'{n:g}' for n in noise_vals]
	# Literal sigma rather than mathtext: Arial carries the glyph, so the whole
	# figure stays in a single PLOS-accepted face
	fig.legend(handles, labels, title='noise amplitude (σ)', loc='lower center',
			   ncol=min(len(noise_vals), 9), fontsize=FONT_SIZE, frameon=False,
			   title_fontsize=FONT_SIZE, handlelength=1.4, columnspacing=1.2,
			   borderaxespad=0.1)

	fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.92,
						wspace=0.15, hspace=0.35)
	_save_(fig, fname)


def figS2_decoding_sweep(fname='figS2_decoding-sweep.pdf'):
	"""Decoding across the sweep.

	A  heatmap of Cohen's d, RNN versus memory-less ANN, over the 9 x 6 design
	B  the same contrast in raw BCE units
	"""
	_set_plos_style_()

	savedict = _load_results_()
	if not _require_(savedict, DECODING_LADDER, 'results.pickle'):
		return

	print('  data present; panels not implemented yet')


def figS3_prediction_reliability(fname='figS3_prediction-reliability.pdf'):
	"""The prediction effect replicates across the five runs.

	The composition stays the unit of analysis (n = 140); the comparison is
	repeated independently within each run rather than on the run average.

	A  paired violins, RNN versus Markov-high, five run-groups side by side
	B  forest plot of Cohen's d per run with bootstrap CI, zero marked
	C  per-run Wilcoxon p on a log axis, with the Bonferroni threshold
	"""
	_set_plos_style_()

	savedict = _load_results_()
	if not _require_(savedict, ['pred_err', 'pred_markh'], 'results.pickle'):
		return

	print('  data present; panels not implemented yet')


def figS4_pe_reliability(fname='figS4_pe-reliability.pdf'):
	"""Prediction-error encoding replicates across the five runs.

	Requires the run axis added to prediction_error_analysis_pipeline(); the
	current pe_results.pickle layout stores a single run.

	A  per-run violins of the two correlations
	B  forest plot of per-run d for baseline versus state-based decoding
	C  per-run p-values with the Bonferroni threshold
	"""
	_set_plos_style_()

	pe_dict = _load_pe_results_()
	if not _require_(pe_dict, ['pe_stm', 'pe_std', 'dec_m', 'dec_b'],
					 'pe_results.pickle'):
		return

	if pe_dict['pe_stm'].ndim < 4:
		print('  pe_results.pickle has no run axis (see prediction_error_analysis_'
			  'pipeline); skipping')
		return

	print('  data present; panels not implemented yet')


# Style and output
def _set_plos_style_():

	# PLOS accepts Arial, Times and Symbol only; fonttype 42 embeds TrueType
	# outlines so that the text stays selectable and editable in the PDF.
	# Called by every figure rather than at import time so that seaborn, which
	# rewrites rcParams when it is imported, cannot win
	mpl.rcParams.update({
		'font.family'     : 'sans-serif',
		'font.sans-serif' : ['Arial', 'Helvetica', 'DejaVu Sans'],
		'font.size'       : FONT_SIZE,
		'axes.labelsize'  : LABEL_SIZE,
		'axes.titlesize'  : LABEL_SIZE,
		'xtick.labelsize' : FONT_SIZE,
		'ytick.labelsize' : FONT_SIZE,
		'legend.fontsize' : FONT_SIZE,
		'mathtext.fontset': 'custom',
		'mathtext.rm'     : 'Arial',
		'mathtext.it'     : 'Arial:italic',
		'mathtext.bf'     : 'Arial:bold',
		'axes.linewidth'  : 0.6,
		'xtick.major.width': 0.6,
		'ytick.major.width': 0.6,
		'xtick.major.size': 2.5,
		'ytick.major.size': 2.5,
		'axes.spines.top' : False,
		'axes.spines.right': False,
		'pdf.fonttype'    : 42,
		'ps.fonttype'     : 42,
		'savefig.dpi'     : 300,
	})


def _save_(fig, fname, results_dir=RESULTS_DIR):

	if fig.get_figheight() > FIG_MAXHEIGHT:
		raise ValueError(f'{fname} is {fig.get_figheight():.2f} in tall; PLOS '
						 f'allows at most {FIG_MAXHEIGHT} in')

	os.makedirs(results_dir, exist_ok=True)
	fpath = os.path.join(results_dir, fname)
	fig.savefig(fpath)
	plt.close(fig)

	print(f'saved {fpath}')


# Data access
def _load_results_(results_path='./results/results.pickle'):

	if not os.path.exists(results_path):
		return None

	return pipeline.load_results(results_path)


def _load_pe_results_(pe_path=PE_RESULTS):

	if not os.path.exists(pe_path):
		return None

	with open(pe_path, 'rb') as f:
		return pickle.load(f)


def _require_(savedict, keys, source):

	# A key that exists but holds nothing but NaN is as good as absent: the sweep
	# is written in full at initialisation and filled in cell by cell
	if savedict is None:
		print(f'  {source} not found; skipping')
		return False

	missing = [k for k in keys
			   if k not in savedict or not np.isfinite(savedict[k]).any()]

	if missing:
		# Labels carry line breaks for the violin axes; flatten them for the log
		labels = ', '.join(MODEL_LABELS.get(k, k).replace('\n', ' ') for k in missing)
		print(f'  no data in {source} for: {labels}; skipping')
		return False

	return True


def _get_cell_(savedict, key, six, nix, run=None):

	# Declared aggregation rule: per-composition test error, averaged across runs
	# unless a single run is requested; the composition is the unit of analysis.
	# Reference-model metrics are stored once per noise level, with no n_hidden or
	# run axis. nanmean allows plotting partially completed sweeps
	arr = savedict[key]

	if arr.ndim == 4:
		return arr[six, nix, run] if run is not None else np.nanmean(arr[six, nix], axis=0)

	return arr[six]


def _available_cells_(savedict, key):

	# (noise index, n_hidden index) pairs holding at least one finite value
	arr = savedict[key]

	if arr.ndim < 4:
		return [(six,) for six in range(arr.shape[0]) if np.isfinite(arr[six]).any()]

	return [(six, nix)
			for six in range(arr.shape[0])
			for nix in range(arr.shape[1])
			if np.isfinite(arr[six, nix]).any()]


def _available_runs_(savedict, key, six, nix):

	arr = savedict[key]

	return [run for run in range(arr.shape[2]) if np.isfinite(arr[six, nix, run]).any()]


# Statistics
def _compute_stats_(x, y=None):

	p_val = ss.wilcoxon(x, y).pvalue

	if y is None:
		cohens_d = x.mean() / x.std()
	else:
		cohens_d = (x.mean() - y.mean()) / ((x.std()**2 + y.std()**2) / 2)**.5

	return cohens_d, p_val


def _bootstrap_ci_(x, y=None, n_boot=2000, ci=95, seed=0):

	# Paired bootstrap over compositions: resample composition indices, recompute
	# Cohen's d. Used for the per-run forest plots, where the interval has to come
	# from the compositions rather than from the five runs
	rng   = np.random.default_rng(seed)
	x     = np.asarray(x)
	idx   = rng.integers(0, len(x), size=(n_boot, len(x)))
	boot  = np.empty(n_boot)

	for b, ix in enumerate(idx):
		boot[b] = _compute_stats_(x[ix], None if y is None else np.asarray(y)[ix])[0]

	half = (100 - ci) / 2

	return np.percentile(boot, [half, 100 - half])


# Colour
def _noise_colors_(noise_vals=None):

	# Colours are assigned from the full planned sweep, so that a partially
	# completed sweep keeps the same colour per noise level across figures.
	#
	# Sequential ramp, light to dark as noise grows: the quietest condition is
	# the one whose loss collapses onto zero and hides against the axis, so it
	# gets the brightest, warmest end. Both ends are truncated - the pale end so
	# that the lightest line still clears 3:1 contrast on white, the dark end so
	# that the ramp stays monotone in luminance
	if noise_vals is None:
		noise_vals = pipeline.NOISE_VALS

	cmap   = plt.get_cmap('plasma')
	steps  = np.linspace(CMAP_LIGHT_END, CMAP_DARK_END, len(noise_vals))
	lookup = {round(n, 6): cmap(s) for n, s in zip(noise_vals, steps)}

	return lambda noise: lookup.get(round(noise, 6), '0.3')


def _model_colors_():

	# Colour encodes one distinction only - RNN versus reference - not which
	# reference. The accent is Okabe-Ito blue (5.2:1 on white); every reference
	# takes the same neutral grey.
	#
	# A four-step grey ramp was the obvious alternative and does not survive the
	# palette validator: between the darkest usable grey and the lightest one that
	# still clears 3:1 contrast there is only room for ~14 OKLab units per step,
	# under the 15 needed to tell adjacent steps apart. Since the references are
	# already separated by labelled position (violins) or dash pattern (lines),
	# spending the colour channel on them buys nothing and invites the reader to
	# hunt for meaning in grey levels that carry none
	# Validated with dataviz/scripts/validate_palette.js: CVD separation 16.1,
	# normal-vision 17.2, both ends over 3:1 on white. The remaining chroma-floor
	# failure is the point - the reference is meant to read as grey
	accent, reference = '#0072b2', '#4d4d4d'

	return {key: accent if key in ('obs_err', 'pred_err') else reference
			for key in MODEL_LABELS}


def _model_linestyles_():

	# Secondary channel for line panels, where every reference shares one grey.
	# Violin panels do not need it: there the ladder position carries identity
	return {
		'obs_err'   : '-',
		'pred_err'  : '-',
		'obs_high'  : '--',
		'pred_low'  : '-',
		'pred_markl': '--',
		'pred_markh': '-.',
		'pred_high' : ':',
	}


# Loss reports (S1)
def _discover_loss_histories_(models_dir=MODELS_DIR):

	# Reports are named by Bachmodel._write_report_; the two training stages of
	# main_fitting() are identified by their objective/freeze combination.
	# Baseline reports carry no _runNN suffix and are excluded by the pattern
	pattern = re.compile(r'^gru_nhidden-(\d+)_nlayers-\d+_chromatic'
						 r'_noise([\dp]+)_obj-(obs|pred)_freeze-.+_run(\d+)\.txt$')

	histories = dict()

	for fpath in sorted(glob.glob(os.path.join(models_dir, '*.txt'))):

		match = pattern.match(os.path.basename(fpath))
		if match is None:
			continue

		n_hidden, noise_str, stage, run = match.groups()
		key = (int(n_hidden), float(noise_str.replace('p', '.')), int(run))

		train_loss = _read_report_(fpath)
		if train_loss.size:
			histories.setdefault(key, dict())[stage] = train_loss

	return histories


def _read_report_(fpath):

	# Line 1 is the per-batch training loss; line 2 (when present) is the
	# periodic validation error, which this figure does not use
	with open(fpath) as f:
		first_line = f.readline().strip()

	if not first_line:
		return np.array([])

	return np.array([float(v) for v in first_line.split(',')])


def _downsample_(x, y, n_points=800):

	# Per-batch losses are far denser than the printed resolution of the figure.
	# A trailing moving average plus subsampling keeps the curve shape while
	# holding the vector PDF to a sane size
	window = max(1, len(y) // (n_points // 2))

	if window > 1:
		y = np.convolve(y, np.ones(window) / window, mode='valid')
		x = x[window - 1:]

	stride = max(1, len(y) // n_points)

	return x[::stride], y[::stride]


def _set_stage_xticks_(ax, obs_len, pred_len):

	# Two or three ticks over the denoising stage: the prediction stage is short
	# in comparison, and a denser grid collides with its end label
	step   = _nice_step_(max(obs_len, 1) / 2)
	ticks  = list(range(0, obs_len + 1, step))
	labels = [f'{t // 1000}k' if t >= 1000 else f'{t}' for t in ticks]

	if pred_len:
		# The stage boundary is already marked by the divider, so its tick is
		# dropped whenever the (much shorter) prediction stage would crowd it
		if pred_len < step:
			ticks, labels = ticks[:-1], labels[:-1]
		end = obs_len + pred_len
		ticks.append(end)
		labels.append(f'{end // 1000}k' if end >= 1000 else f'{end}')

	ax.set_xticks(ticks)
	ax.set_xticklabels(labels)


def _nice_step_(target):

	exponent = 10 ** np.floor(np.log10(target))
	mantissa = min([1, 2, 5, 10], key=lambda m: abs(m - target / exponent))

	return max(int(mantissa * exponent), 1)


# Annotation
def _plot_point_lines_(ax, data, col0, col1, x0, x1):

	options = {'color': 'k', 'linewidth': 0.1, 'alpha': 0.2}

	for y0, y1 in zip(data[col0], data[col1]):
		ax.plot([x0, x1], [y0, y1], **options)


def _write_stats_pair_(ax, data, col0, col1, x0, x1, y, up=True):

	cohens_d, p_val = _compute_stats_(data[col0], data[col1])

	y0, y1 = y, y + (1 if up else -1) * 0.02
	text = f'd = {cohens_d:.2f} (p = {p_val:.1g})'

	ax.plot([x0, x0], [y0, y1], 'k-', linewidth=0.7)
	ax.plot([x0, x1], [y1, y1], 'k-', linewidth=0.7)
	ax.plot([x1, x1], [y1, y0], 'k-', linewidth=0.7)

	sign_options = {'fontsize': FONT_SIZE - 1,
					'horizontalalignment': 'center',
					'verticalalignment': 'center'}

	ax.text((x0 + x1) / 2, y + (1 if up else -1) * 0.04, text, **sign_options)


def _write_stats_point_(ax, data, col, x, y):

	cohens_d, p_val = _compute_stats_(data[col])

	sign_options = {'fontsize': FONT_SIZE - 1,
					'horizontalalignment': 'center',
					'verticalalignment': 'center'}

	ax.text(x, y, f'd = {cohens_d:.2f} (p = {p_val:.1g})', **sign_options)


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

	return f'{val_str} ({pval_str})'


# Panel builders, to be filled in alongside the figures that consume them. Their
# signatures depend on the array layouts they will read, so they are declared
# rather than guessed at
def _ladder_violins_(ax, savedict, keys, six, nix, run=None):
	"""Violins of per-composition error for an ordered ladder of models."""
	pass


def _forest_plot_(ax, effects, intervals, labels):
	"""Cohen's d per run with bootstrap CIs and a zero reference line."""
	pass


def _heatmap_(ax, values, annot, vmax, xlabel, ylabel):
	"""Noise x n_hidden map of an effect size or a raw difference."""
	pass


def _piano_roll_(ax, array, cmap='Greys'):
	"""12 x time binary or probabilistic chord matrix."""
	pass


def _load_example_model_(noise=EXAMPLE_NOISE, n_hidden=EXAMPLE_NHIDDEN, run=EXAMPLE_RUN):
	"""Load one trained network plus a test Chunker for the example panels.

	Fig 1C and 2A need the per-timestep obs/pred outputs, which
	Bachmodel.compute_pe_and_state does not return - it yields only pe, stm and
	std. Call m.net(sample) directly under torch.no_grad() here rather than
	widening the library API.
	"""
	pass


def make_all_figures():

	# Main figures are figNN_<name>, supplementary ones figSN_<name>; sorting puts
	# the numbered ones first because '0' precedes 'S'
	is_figure = re.compile(r'^fig(S?\d{1,2})_').match

	figure_funcs = sorted(name for name in globals()
						  if is_figure(name) and callable(globals()[name]))

	for name in figure_funcs:
		print(f'\n{name}')
		print('-' * len(name))
		globals()[name]()


if __name__ == '__main__':

	make_all_figures()
