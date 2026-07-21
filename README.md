
# Bayes and Bach: An RNN-powered affair #

This project studies whether recurrent neural networks trained to denoise inputs with complex temporal correlations spontaneously learn to predict the next input to increase accuracy. The project uses Back compositions as inputs, so that the undelying generative model is structured and complex.

This project kickstarted at [BrainHack Donostia 2024](https://brainhack-donostia.github.io) and is currently under development.

## Code ##

### `bachbayes.py` ###

Core library, holding every class the rest of the code builds on:

- `RNN` and `FeedForwardNN`: the two `torch` architectures. The recurrent one (Elman/GRU/LSTM) carries two sigmoid readouts, one for the denoised observation and one for the prediction of the next input; the feedforward one is memory-less and serves as a baseline.
- `Bachmodel`: wraps a network together with its parameters (noise level, hidden units, chromatic vs. full-range encoding). Handles training (with per-layer freezing, selectable `obs`/`pred` objectives and early stopping on the validation set), testing, weight saving/loading, and extraction of prediction errors and hidden-state trajectories.
- `Chunker`: data loader. Reads the `.csv` scores into binary piano-roll arrays (12 chromatic pitch classes or 108 MIDI notes), adds Gaussian noise to build the network's input, and serves either random training chunks or whole pieces.
- `BWVRetriever`: scraper that crawls public Bach MIDI archives, matches files to their BWV number, parses the MIDI into chord-per-tick target lists, and stores them as `.csv` in `data/`.

### `pipeline.py` ###

Driver script that runs the experiments end to end:

- `prepare_data()`: downloads the corpus with `BWVRetriever` and splits it into `training/`, `validation/` and `test/`.
- `main_fitting()`: trains a GRU first on denoising (prediction readout frozen), then trains the prediction readout on top of the frozen recurrent weights, and reports observation and prediction errors.
- `baselines()`: fits the reference models used as upper/lower bounds (memory-less ANN, global note distribution, first-order Markov with noisy or ground-truth input, and a GRU trained directly to predict).
- `pipeline()`: sweeps noise levels × hidden-unit counts × runs, calling the above and storing results in `results/results.pickle`.
- `prediction_error_analysis_pipeline()`: for each fitted model, correlates prediction error with hidden-state magnitude and its derivative, and measures how well a linear decoder recovers the prediction error from network activity.

### `figures.py` ###

Plotting layer over the pickled results. Produces the violin plots comparing the RNN against each baseline for a single model (`plot_performance_example`, `plot_prederr_example`), the example time courses and correlations for one test piece (`plot_prederr_example_opera`), and the noise × hidden-units heatmaps of effect sizes and raw differences (`plot_perfomance_summary`, `plot_prederr_summary`). Private helpers handle Wilcoxon tests, Cohen's *d*, significance brackets and annotation formatting. Figures are written to `results/` as PDFs.
