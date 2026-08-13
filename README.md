
# Bayes and Bach: An RNN-powered affair #

This project studies whether recurrent neural networks trained to denoise inputs with complex temporal correlations spontaneously learn to predict the next input to increase accuracy. The project uses Bach compositions as inputs, so that the underlying generative model is structured and complex.

This project kickstarted at [BrainHack Donostia 2024](https://brainhack-donostia.github.io) and is currently under development.

## Code ##

### `bachbayes.py` ###

Core library, holding every class the rest of the code builds on:

- `RNN` and `FeedForwardNN`: the two `torch` architectures. The recurrent one (Elman/GRU) carries two sigmoid readouts, one for the denoised observation and one for the prediction of the next input; the feedforward one is memory-less and serves as a baseline.
- `Bachmodel`: wraps a network together with its parameters (noise level, hidden units, chromatic vs. full-range encoding). Handles training (with per-layer freezing, selectable `obs`/`pred` objectives, and a fixed batch budget with periodic validation and best-checkpoint selection), testing, weight saving/loading, and extraction of prediction errors and hidden-state trajectories.
- `LogisticModel`: the memory-less reference. Each output class is an independent logistic regression on the whole input vector at a single time step, so there is no memory and no hidden layer. `obj` and `target_as_input` pick which quantity it estimates, exactly as for `Bachmodel`: the observation decoder (`x_t → y_t`), the first-order Markov predictor from noisy input (`x_t → y_{t+1}`), or the same from clean input (`y_t → y_{t+1}`). It replaced the feedforward reference nets, whose non-convex fits diverged to a saturated constant output at high noise; the logistic fit is convex and can never do worse than the marginal predictor, which is the correct floor for a memory-less bound. Its public interface mirrors the parts of `Bachmodel` that `pipeline.baselines()` uses, so the driver treats it like any other model.
- `Chunker`: data loader. Reads the `.csv` scores into binary piano-roll arrays (12 chromatic pitch classes or 108 MIDI notes) and serves either random training chunks (with fresh Gaussian noise drawn on every chunk) or whole pieces (with a fixed noise realisation, for validation/testing).
- `BWVRetriever`: scraper that crawls public Bach MIDI archives, matches files to their BWV number, parses the MIDI into chord-per-tick target lists, and stores them as `.csv` in `data/`.

### `pipeline.py` ###

Driver script that runs the experiments end to end:

- `prepare_data()`: downloads the corpus with `BWVRetriever` and splits it into `training/`, `validation/` and `test/`.
- `main_fitting()`: trains a GRU first on denoising (prediction readout frozen), then trains the prediction readout on top of the frozen recurrent weights, and reports observation and prediction errors.
- `baselines()`: fits the reference models used as upper/lower bounds — the memory-less logistic decoder, the global note distribution, first-order Markov with noisy or ground-truth input, and a GRU trained directly to predict. These estimate properties of the task rather than of any one network, so they are computed once per noise level at saturated capacity instead of being width-matched to each RNN of the sweep.
- `pipeline()` and `baselines_pipeline()`: sweep noise levels × hidden-unit counts × runs, calling the above and storing results in `results/results.pickle`. Both are resumable — finished cells are checkpointed on disk and reloaded rather than refitted, so an interrupted sweep continues where it stopped.
- `prediction_error_analysis_pipeline()`: for each fitted model, correlates the innovation magnitude with the state-update magnitude, then decodes each prediction-error target from a constant, from the input alone, and from the hidden state. Writes `results/pe_results.pickle` with `pe_std` plus one entry per target × decoder pair (`dec_{raw,sig,neg}_{b,x,h}`), resuming per cell.

### `figures.py` ###

Figure factory for the manuscript. Every `figNN_*` (main) and `figSN_*` (supplementary) function produces one publication-ready PDF under `results/`; running the module executes all of them in name order via `make_all_figures()`, so the figure number lives in the function name. The Results section presents two effects, each shown first in a single network and then across the noise × network-size sweep:

| | |
|---|---|
| Fig 1 | task, architecture, and evidence the denoising task was solved |
| Fig 2 | prediction emerges — single network |
| Fig 3 | prediction emerges — across the sweep |
| Fig 4 | prediction errors are encoded — single network |
| Fig 5 | prediction errors are encoded — across the sweep |
| S1 | training loss timecourses |
| S2 | decoding across the sweep |
| S3 | the prediction effect replicates across the five runs |
| S4 | PE encoding replicates across the five runs |

The prediction-error figures (4, 5, S4) read `results/pe_results.pickle`. Their design answers the input-drive confound: the state necessarily encodes the observation it has just consumed, so PE encoding is only claimed where a state decoder beats an input-only control — most stringently for the omission component, which the current input cannot carry. Every figure guards on what it needs and plots only what is present, so the module stays runnable while any analysis output is missing. All figures are laid out to PLOS Computational Biology specifications: full column width (7.5 in), Arial at 8–10 pt, fonts embedded as TrueType. The PDFs are build products and are not tracked.

### `run_pipeline.sh` ###

Restart wrapper: reruns `pipeline.py` on failure, up to `MAX_ATTEMPTS` times with `PAUSE` seconds between tries. A shell loop rather than a `try`/`except` inside Python because the crashes it guards against are intermittent CUDA/cuDNN faults, and a CUDA error can leave the process's CUDA context corrupted — retrying in the *same* process would just keep failing. A fresh interpreter each attempt gets a clean context, and since the pipeline is resumable, a restart continues from where the crash happened rather than redoing work.
