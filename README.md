![Momenta — Exploring real and synthetic speech through audio features.](docs/assets/banner.svg)

# Momenta · Audio authenticity experiments

A research notebook exploring real-versus-synthetic speech classification with acoustic features, classical machine learning and neural networks.

[Notebook](momenta.ipynb) · [Analysis report](Documentation%20and%20analysis.pdf) · [Dataset pointer](DATASET) · [Original notes](README-ORIGINAL.md)

## Experiment design

| Branch | Features | Model family |
|---|---|---|
| Spectral / pitch | Mean MFCC coefficients and pitch-track statistics | Convolutional layers and bidirectional LSTMs |
| Energy | Mean root-mean-square energy | Logistic regression |
| Signal crossings | Mean zero-crossing rate | Dense neural classifier |

The notebook loads separate training, validation and testing folders. Labels are `0` for fake and `1` for real. It explores imputation, scaling, training-set oversampling, class weighting and decision thresholds, and reports classification metrics and confusion matrices.

## Files

| File | Purpose |
|---|---|
| `momenta.ipynb` | Experimental implementation and saved outputs |
| `Documentation and analysis.pdf` | Existing project report |
| `DATASET` | Original dataset download pointer |
| `README-ORIGINAL.md` | Preserved dependency notes |

## Prepare the environment

Use an isolated Python environment with Jupyter, NumPy, pandas, librosa, TensorFlow, scikit-learn, imbalanced-learn and tqdm. Package versions were not pinned in the original project; this is an experimental dependency list, not a verified lockfile.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab numpy pandas librosa tensorflow scikit-learn imbalanced-learn tqdm
jupyter lab momenta.ipynb
```

Download and extract the dataset referenced in `DATASET`. Update `base_dir` in the notebook to your extraction directory:

```text
for-original/
  training/fake/      training/real/
  validation/fake/    validation/real/
  testing/fake/       testing/real/
```

## Reproducibility notes

The existing notebook is **not yet a clean “Run All” pipeline**. Source inspection found `SMOTE` used before its later import, `compute_class_weight` used without a corresponding import, and `best_threshold` referenced before its later definition. Resolve those cell-order dependencies before training. The CNN input shape is hard-coded to 1038 feature positions; verify that your librosa configuration produces the expected shape.

Threshold selection must use validation data. The notebook also explores a test-optimal threshold; results selected on the test set are exploratory and should not be presented as untouched holdout performance. No new training run or performance claim is introduced by this documentation update.

## Reading the results

Start with the analysis PDF, then inspect the matching notebook cells and their preprocessing. Compare precision, recall and F1 alongside accuracy and the confusion matrix. Keep preprocessing fit on training data and record dataset version, split counts, random seeds, package versions and threshold-selection procedure in any reproduction.

## Project status

Research artifact with documented reproduction work remaining. The original notebook and report are retained. No new software or dataset license is assigned by this documentation refresh.
