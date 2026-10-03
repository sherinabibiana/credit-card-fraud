# Credit Card Fraud Detection — MLOps with Airflow + MLflow

An end-to-end, locally verified ML pipeline for the ULB/Worldline credit card
fraud dataset (OpenML dataset **1597**). Apache Airflow orchestrates the work,
MLflow records every model run, and scikit-learn does the modelling.

Everything below was executed and verified on this machine — no numbers in this
README are illustrative.

---

## Results (verified)

| model | precision | recall | F1 | ROC-AUC | avg precision |
|---|---|---|---|---|---|
| Logistic Regression | 0.0653 | 0.9053 | 0.1218 | 0.9826 | 0.6920 |
| **Random Forest** | **0.8315** | **0.7789** | **0.8043** | 0.9784 | **0.7911** |

* Split: **220,530 train rows** (378 fraud) / **55,133 test rows** (95 fraud)
* Selection criterion: highest `average_precision`, tie-break `f1_score`
* Selected model: **Random Forest** — `0.7911` avg precision vs `0.6920` for LR
* Confusion matrix, RF: TP=74, FP=15, FN=21, TN=55,023
* Both runs recorded in MLflow with `trigger=airflow_dag`, `dag_id=credit_card_fraud_detection`

> These numbers describe **one stratified split of one public dataset**. They are
> an educational demonstration, not evidence of production readiness. With only
> 95 fraud rows in the test set, differences of a few points are not meaningful.

### Why average precision and not accuracy?

Fraud is 0.17% of the data. A model that predicts "not fraud" for everything
scores **99.83% accuracy** while catching zero fraud. Accuracy is therefore
useless here; average precision (PR-AUC) is the primary criterion.

### Why class weights and not SMOTE?

`class_weight="balanced"` (LR) and `class_weight="balanced_subsample"` (RF) fix
the imbalance without generating synthetic rows. This keeps the test set free of
synthetic neighbours that could inflate scores, and keeps the pipeline
deterministic. This is a defensible default, not the only valid choice.

---

## Feature decisions

**`Time` is dropped by default.** It is "seconds since the first transaction in
the file", so it encodes *row order* rather than transaction content — a model
can key on it and score well while learning nothing transferable. Set
`FRAUD_DROP_TIME=false` to keep it and compare.

**Exact duplicate model inputs are removed.** The raw file has 1,081 fully
duplicated rows. After dropping `Time`, 8,063 rows share an identical
`V1..V28 + Amount` feature vector with another row. Left in place they can land
in *both* train and test, which is leakage. The pipeline de-duplicates on the
exact model inputs, keeping the first occurrence:

| stage | rows | fraud |
|---|---|---|
| raw | 284,807 | 492 |
| after dropping `Time` + de-duplication | 275,663 | 473 |

A hard assertion verifies train and test share **0 identical rows**.

---

## Layout

```
credit-card-fraud-mlops/
├── dags/fraud_detection_dag.py   7-task Airflow DAG
├── src/
│   ├── config.py                 all paths, thresholds, model params
│   ├── data_loader.py            OpenML download (parquet → ARFF) + cache
│   ├── data_validation.py        hard-fail validation → JSON report
│   ├── preprocessing.py          dedup, stratified split, sklearn preprocessing
│   ├── train.py                  LogisticRegression + RandomForest pipelines
│   ├── evaluate.py               metrics, plots, model selection
│   ├── mlflow_utils.py           experiment, runs, artifacts, model logging
│   ├── exceptions.py             typed errors
│   └── logging_utils.py          consistent logging
├── scripts/
│   ├── env.sh                    shared environment (source this first)
│   ├── download_data.py          fetch + cache the dataset
│   ├── run_pipeline.py           full pipeline without Airflow
│   └── start_airflow.sh          start/stop Airflow, wait until healthy
├── tests/                        77 tests
├── data/{raw,interim}/           cached dataset, split hand-off files
├── models/                       fitted *.joblib pipelines
├── artifacts/<model>/            plots, metrics, predictions, reports
└── reports/                      validation + pipeline summaries (JSON)
```

---

## Requirements

Everything runs inside **WSL2 (Ubuntu 24.04)** using the virtualenv that already
exists on this machine — `~/.venvs/houseprice`. No extra installation is needed.

```
Python 3.12.3
apache-airflow 3.3.2   mlflow 3.16.1        scikit-learn 1.9.1
pandas 3.0.5           numpy 2.5.3         scipy 1.18.1
matplotlib 3.11.2      pyarrow 25.0.1      pytest 9.1.1
```

For a *fresh* environment, pin Airflow with its constraints file (replace `3.12`
with your Python minor version):

```bash
pip install -r requirements.txt \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.3.2/constraints-3.12.txt"
```

---

## Quick start

Every command below runs in WSL. `source scripts/env.sh` sets the venv, ports,
`AIRFLOW_HOME` (kept **inside** the project at `.airflow/`) and `PYTHONPATH`.

```bash
cd /mnt/e/Data\ Science/Projects/credit-card-fraud-mlops
source scripts/env.sh
```

### 1. Fetch the dataset (optional — the pipeline does this for you)

```bash
python -m scripts.download_data
```

Source: <https://www.openml.org/d/1597> — parquet mirror first, MD5-verified
ARFF as fallback, then cached to `data/raw/creditcard.csv`.

### 2. Run the pipeline without Airflow

```bash
python -m scripts.run_pipeline                # logs to MLflow
python -m scripts.run_pipeline --no-mlflow    # train/evaluate only
python -m scripts.run_pipeline --force-download
```

### 3. Run it through Airflow

```bash
bash scripts/start_airflow.sh          # starts, waits for healthy, prints admin pw
bash scripts/start_airflow.sh stop     # stop it
```

Then open <http://localhost:8080>, sign in as `admin` with the password printed
by the script (stored in `.airflow/simple_auth_manager_passwords.json.generated`),
and press **Trigger DAG** on `credit_card_fraud_detection`. The DAG is created
paused, so unpause it once via the UI or:

```bash
airflow dags unpause credit_card_fraud_detection
airflow dags trigger  credit_card_fraud_detection
```

### 4. Inspect the results

| what | where |
|---|---|
| Airflow UI | <http://localhost:8080> |
| MLflow UI | <http://localhost:5000> — experiment `credit-card-fraud-detection` |
| Task logs | Airflow UI → DAG → task → Log |
| Plots/metrics | `artifacts/<model>/` |
| Summaries | `reports/airflow_pipeline_summary.json`, `reports/validation_report.json` |

### 5. Run the tests

```bash
python -m pytest -q
# 77 passed
```

---

## The DAG

```
load_data → validate_data → prepare_data → split_data
          → train_models → evaluate_models → log_results
```

Each task is a thin wrapper: the real work lives in `src/`, so the same code
path runs from the Airflow UI, from `python -m scripts.run_pipeline`, and from
the tests. Airflow owns ordering, retries (1) and per-task logs; MLflow owns the
experiment history.

Verified task timings for one run (220,530 training rows):

| task | duration |
|---|---|
| load_data | ~8 s |
| validate_data | ~8 s |
| prepare_data | ~23 s |
| split_data | ~16 s |
| train_models | ~49 s (RF 36 s) |
| evaluate_models | ~26 s |
| log_results | ~2 s |

### Two bugs worth knowing about

Both were found by running the DAG, not by reading the code, and both now have
regression tests:

1. **`@task` name shadowing.** The task `split_data` rebound the name that the
   module had imported from `src.preprocessing`, so the task body called the
   task object instead of the function — failing at runtime with
   `TypeError: too many positional arguments`. The import is now aliased
   (`split_dataset`), and `tests/test_dag.py` asserts that no task body
   references a name that is also a task id.

2. **Scoring on the training set.** `X_test, y_test, _, _, _ = load_split()`
   bound the *training* partition to those names, because `load_split()` returns
   `(X_train, y_train, X_test, y_test, manifest)`. The run still produced
   confident-looking metrics — RF recall of 1.0, on 220,494 rows instead of
   55,133. `evaluate_models` now unpacks all five values explicitly and asserts
   the row and fraud counts match the split manifest before scoring.

---

## Configuration

All settings live in `src/config.py` and can be overridden by environment
variables (see `.env.example`).

| variable | default | meaning |
|---|---|---|
| `FRAUD_DROP_TIME` | `true` | drop the `Time` column |
| `FRAUD_RANDOM_STATE` | `42` | seed for split and estimators |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | tracking server |
| `MLFLOW_EXPERIMENT_NAME` | `credit-card-fraud-detection` | experiment |
| `AIRFLOW_PORT` / `MLFLOW_PORT` | `8080` / `5000` | UI ports |

Validation thresholds (in `src/config.py`): minimum 1,000 rows, maximum 5%
missing values. A breach raises `DataValidationError` and fails the DAG rather
than training on bad data.

### MLflow 3 and skops

MLflow 3 serialises scikit-learn models with `skops`, which refuses to persist
types it does not recognise. `MLFLOW_SKOPS_TRUSTED_TYPES` in `src/config.py`
trusts `sklearn.tree._tree.Tree` (the C array behind tree ensembles) and
`numpy.dtype` (column dtypes carried by the input example).

---

## Notes

* `artifacts/**/predictions.csv` is capped at 20,000 rows via
  `PREDICTION_DUMP_LIMIT`; the metrics themselves always use the full test set.
* When reading MLflow through the REST API, use the `artifacts/list` endpoint —
  the `artifacts` array inside `runs/search` and `runs/get` responses comes back
  empty in MLflow 3 even when artifacts exist.
* Airflow 3's health endpoint is `/api/v2/monitor/health`, not `/health`.
* `airflow dags list-runs <dag_id> -o json` prints Airflow's missing-`graphviz`
  warning to **stdout** before the JSON. Prefix with `PYTHONWARNINGS=ignore` or
  slice from the first `[` when parsing.

## Data source

ULB/Worldline credit card fraud dataset, via OpenML dataset 1597.
<https://www.openml.org/d/1597> — two days of transactions, 284,807 records,
30 features plus `Class`.