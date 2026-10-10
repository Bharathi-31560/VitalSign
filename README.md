# VitalSign - Vital Sign Analysis & Prediction System

A complete Flask + scikit-learn web application that explores a patient dataset, preprocesses it
without data leakage, trains and compares several machine-learning models, and predicts the
outcome for a new patient through a web form.

## Important: what the dataset actually contains

The file in `data/vital_sign.csv` (101,766 rows x 50 columns) is a **hospital-encounter dataset of
diabetic patients** (the columns match the public *Diabetes 130-US hospitals* data). It contains
demographics, length of stay, number of lab procedures / medications / prior visits, ICD-9
diagnoses, lab-test results and 23 diabetes-drug columns. It does **not** contain bedside
measurements such as heart rate, SpO2 or blood pressure. The whole project was therefore built
around the columns that really exist - nothing was invented.

### Target column (decision)

The dataset's own outcome column is `readmitted` with values `NO`, `>30`, `<30`.
**The project predicts whether a patient is readmitted at all**: `>30` or `<30` -> `Readmitted (1)`,
`NO` -> `Not readmitted (0)`. Reasons: it is the dataset's obvious outcome, and the binary form is
reasonably balanced (about 41% readmitted after cleaning), which gives meaningful precision,
recall and ROC-AUC. To predict only 30-day readmission instead, change the one line
`d[TARGET] = (d[TARGET_RAW] != "NO")` in `src/data/preprocess.py` to `== "<30"` and re-run
`python main.py --train`.

### Cleaning decisions (all visible on the Preprocessing page)

| Step | Why |
|---|---|
| Remove 3 rows with gender "Unknown/Invalid" | invalid value |
| Remove 2,423 encounters discharged to hospice / expired | these patients cannot be readmitted |
| Keep only the **first encounter per patient** | otherwise the same patient lands in both train and test sets (leakage) |
| Drop `encounter_id`, `patient_nbr` | identifiers |
| Drop `weight` (96% missing) and constant drug columns | no information |
| Fill missing categorical values with `Unknown` | `payer_code`, `medical_specialty`, `race` gaps can be informative |
| Group ICD-9 codes in `diag_1/2/3` into clinical groups | 700+ raw codes are too sparse |
| `age` band -> numeric mid-point | ordered variable |
| Treat admission/discharge codes as categories | they are codes, not quantities |

## Features

* Dashboard with live statistics from the dataset and the model results
* Dataset browser (paginated, column types, missing values, duplicates)
* EDA: histograms, box plots, count plots, scatter plots, correlation heatmap, distribution plots,
  pair plot, target distribution, missing values - all generated from the real data
* Preprocessing page with before/after comparison and a leakage-safe workflow
* 7 supervised models + K-Means (with elbow method) + DBSCAN
* Evaluation (accuracy, precision, recall, F1, confusion matrix, ROC-AUC), comparison table & charts
* Prediction form generated automatically from the dataset columns; uses the same saved pipeline as training
* Friendly error pages (missing dataset, missing/corrupt model, invalid input)

## Folder structure

```
VitalSign/
├── app/
│   ├── app.py                  <- Flask application (RUN THIS FILE)
│   ├── static/
│   │   ├── charts/             <- generated chart images
│   │   └── css/style.css
│   └── templates/              <- base, home, load_dataset, eda, Preprocessing, models,
│                                  model_detail, evaluation, comparison, prediction, error
├── data/vital_sign.csv         <- your dataset
├── models/                     <- trained pipelines (.pkl), preprocessing_pipeline.pkl, result JSON files
├── src/
│   ├── config.py               <- paths and constants (relative paths only)
│   ├── ml_utils.py             <- shared training / evaluation / chart helpers
│   ├── model_registry.py       <- loads the files in "ML models"
│   ├── pipeline.py             <- runs EDA -> preprocessing -> all models -> comparison
│   ├── predict.py              <- input validation + prediction
│   ├── data/                   <- load_data.py, eda.py, preprocess.py
│   └── ML models/              <- logistic_regression, decision_tree, random_forest, gradient_boost,
│                                  adaboost, knn, svm, kmeans, kmeans_elbow, dbscan
├── main.py                     <- optional launcher (and `--train`)
├── requirements.txt
└── README.md
```

## Installation

Requires Python 3.9 or newer.

```bash
python -m venv .venv
```

Activate it - Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install the packages:

```bash
pip install -r requirements.txt
```

## Run

```bash
python app/app.py
```

Then open **http://127.0.0.1:5000** in your browser. (`python main.py` does the same.)

In PyCharm: open the `VitalSign` folder, select the `.venv` interpreter, install `requirements.txt`,
right-click `app/app.py` -> *Run 'app'*.

The ZIP already contains trained models and charts, so the app works immediately. To regenerate
everything (about 2-4 minutes) either click **Run analysis & train models** in the app (shown if
outputs are missing), or run:

```bash
python main.py --train
```

If you use a very different scikit-learn version than the one used to create the ZIP and a model fails to
load, simply run the command above to retrain with your version.

## Workflows

**EDA** (`src/data/eda.py`): statistics (describe, outliers by 1.5 x IQR, correlations, unique values) and
charts are produced from the cleaned data and saved to `app/static/charts/`.

**Preprocessing** (`src/data/preprocess.py`): load -> rule-based cleaning -> separate X and y ->
stratified 80/20 split -> `ColumnTransformer` fitted on the training set only (median imputation +
`StandardScaler` for numbers; "Unknown" imputation + `OneHotEncoder` for categories; categories under 1% grouped) ->
transform train and test.

**ML** (`src/ML models/`): each file defines `INFO` (description shown in the app) and `run(data)`, which
trains the model, evaluates it on the test set, saves a full pipeline (preprocessing + model) to `models/`
and returns metrics. Shared logic lives in `src/ml_utils.py`. KNN and SVM are trained on stratified
subsamples (15,000 and 8,000 rows) because they scale poorly; they are still tested on the full test set.

**Best model**: chosen by ROC-AUC (ties by F1) and copied to `models/best_model.pkl`.

**Prediction**: the form is built from `models/meta.json` (columns, ranges, options). Input is validated,
turned into a one-row DataFrame and passed through the saved pipeline - the same preprocessing as training.

## Models

Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, KNN, SVM (supervised
classification) and K-Means (elbow + silhouette) and DBSCAN (k-distance plot for eps) as unsupervised
analysis. Readmission is a hard problem with this data: results are around 62-63% accuracy and 0.62-0.66
ROC-AUC. These are the real, measured results - see the Evaluation and Comparison pages.

## Disclaimer

Educational project. Predictions are statistical estimates, not medical advice.
