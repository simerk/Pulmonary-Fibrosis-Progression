# Pulmonary Fibrosis Progression

A machine learning project for the [OSIC Pulmonary Fibrosis Progression](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression) Kaggle competition. The goal is to predict a patient's lung function decline (FVC — Forced Vital Capacity) from a baseline CT scan and clinical metadata, along with a confidence estimate for each prediction.

## Notebooks

| Notebook | Description |
|---|---|
| `eda-minor.ipynb` | Exploratory data analysis: clinical metadata distributions (Age, Sex, Smoking Status, FVC, Weeks), DICOM CT scan loading, Hounsfield Unit conversion, and lung segmentation/masking. |
| `minor-pulmonary-fibrosis-project.ipynb` | Modeling pipeline: <br>1. **EfficientNet CNN** — predicts a per-patient FVC decline coefficient from CT images + tabular features. <br>2. **Quantile regression (Keras)** — predicts FVC quantiles (q25/q50/q75) from tabular features, trained with a custom competition metric (Laplace log-likelihood). <br>3. **Ensemble** — blends the CNN and quantile regression outputs to produce the final submission. |

## Approach

- **Tabular features**: normalized age, sex, one-hot smoking status, baseline FVC, and week offset from baseline.
- **Image features**: CT slices read via `pydicom`, resized and fed through an EfficientNet backbone.
- **Target**: FVC value plus a confidence/uncertainty estimate, scored with the competition's clipped Laplace log-likelihood metric.
- **Validation**: K-Fold cross-validation on the quantile regression model.

## Setup

```bash
pip install -r requirements.txt
```

Data is expected under `input/osic-pulmonary-fibrosis-progression/` (train.csv, test.csv, sample_submission.csv, and train/test DICOM folders), matching the Kaggle competition's directory layout. Data is **not** included in this repo — download it from Kaggle and place it locally.

## Status

Exploratory / competition notebooks — not yet refactored into a reusable package. See notebooks for details; some cells are incomplete or exploratory.
