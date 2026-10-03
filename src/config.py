"""Central, machine-independent configuration.

Every path is derived from the location of this file, so the project works no
matter where it is checked out and no absolute path is ever hardcoded.

Values can be overridden through environment variables (optionally loaded from
a local ``.env`` file - see ``.env.example``).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Optional local overrides; silently does nothing when the file is absent.
load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean flag from the environment."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Read an integer from the environment, falling back on bad input."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# --------------------------------------------------------------------------
# Directories
# --------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"

MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Cached copy of the dataset produced by the loader (never committed).
CACHED_DATASET_PATH = RAW_DATA_DIR / "creditcard.csv"

# Hand-off files between Airflow tasks.
PREPARED_DATA_PATH = INTERIM_DATA_DIR / "prepared.csv"
TRAIN_DATA_PATH = INTERIM_DATA_DIR / "train.csv"
TEST_DATA_PATH = INTERIM_DATA_DIR / "test.csv"
SPLIT_MANIFEST_PATH = INTERIM_DATA_DIR / "split_manifest.json"
TRAINING_REPORT_PATH = INTERIM_DATA_DIR / "training_report.json"
VALIDATION_REPORT_PATH = REPORTS_DIR / "validation_report.json"

# --------------------------------------------------------------------------
# Dataset source (OpenML, dataset id 1597 = the ULB/Worldline credit card
# fraud dataset). Two independent mirrors are configured so the pipeline can
# still run if one endpoint is unavailable.
# --------------------------------------------------------------------------
DATASET_NAME = "credit-card-fraud-detection"
OPENML_DATASET_ID = 1597
OPENML_PARQUET_URL = "https://data.openml.org/datasets/0000/1597/dataset_1597.pq"
OPENML_ARFF_URL = "https://openml.org/data/v1/download/1673544/creditcard.arff"
OPENML_ARFF_MD5 = "178bcf9bb1f31a3dfe12d0e577884add"
DOWNLOAD_TIMEOUT_SECONDS = 600

# Schema of the dataset as published on OpenML (verified against the download).
TARGET_COLUMN = "Class"
LEGITIMATE_LABEL = 0
FRAUD_LABEL = 1
EXPECTED_COLUMNS: tuple[str, ...] = (
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount",
    "Class",
)

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_STATE = _env_int("FRAUD_RANDOM_STATE", 42)
TEST_SIZE = 0.2

# --------------------------------------------------------------------------
# Feature engineering decisions (documented in the README)
# --------------------------------------------------------------------------
# 'Time' is the number of seconds since the first transaction in the file. It
# encodes row order, not transaction content, so it is dropped by default to
# stop the model keying on position-in-file. Set FRAUD_DROP_TIME=false to keep
# it and compare.
DROP_TIME = _env_bool("FRAUD_DROP_TIME", True)
TIME_COLUMN = "Time"
AMOUNT_COLUMN = "Amount"

# --------------------------------------------------------------------------
# Validation thresholds - a breach raises DataValidationError and fails the DAG
# --------------------------------------------------------------------------
MIN_ROWS = 1_000
MAX_MISSING_RATIO = 0.05
ALLOW_DUPLICATE_ROWS = True

# --------------------------------------------------------------------------
# Models. Only hyperparameters that genuinely exist on the estimator are
# logged to MLflow.
# --------------------------------------------------------------------------
LOGISTIC_REGRESSION_PARAMS: dict[str, object] = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

RANDOM_FOREST_PARAMS: dict[str, object] = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_leaf": 1,
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

MODEL_NAMES: tuple[str, ...] = ("logistic_regression", "random_forest")

# Scale the numeric features only for distance/gradient based models; a random
# forest is invariant to feature scale, so scaling is skipped for it.
SCALE_FOR_MODELS: tuple[str, ...] = ("logistic_regression",)
IMPUTE_STRATEGY = "median"

# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
# Documented, non-arbitrary selection criterion (see README):
#   primary  = average_precision (PR-AUC, the meaningful metric under heavy
#              imbalance), tie-break = f1_score.
MODEL_SELECTION_PRIMARY_METRIC = "average_precision"
MODEL_SELECTION_TIEBREAK_METRIC = "f1_score"
PREDICTION_DUMP_LIMIT = 20_000

# --------------------------------------------------------------------------
# MLflow
# --------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME", "credit-card-fraud-detection"
)
MLFLOW_ARTIFACT_DIRNAME = "plots"
# MLflow 3 serialises sklearn models with skops, which refuses to persist types
# it does not recognise unless the owning code explicitly trusts them.
#   - sklearn.tree._tree.Tree : the raw C array behind every tree ensemble
#   - numpy.dtype             : column dtypes carried by the input_example
MLFLOW_SKOPS_TRUSTED_TYPES = [
    "sklearn.tree._tree.Tree",
    "numpy.dtype",
]

# --------------------------------------------------------------------------
# Airflow (consumed by scripts/env.sh and the DAG)
# --------------------------------------------------------------------------
AIRFLOW_HOME = PROJECT_ROOT / ".airflow"
DAGS_DIR = PROJECT_ROOT / "dags"
AIRFLOW_PORT = _env_int("AIRFLOW_PORT", 8080)
MLFLOW_PORT = _env_int("MLFLOW_PORT", 5000)


def ensure_directories() -> None:
    """Create every directory the pipeline writes to. Safe to call repeatedly."""
    for directory in (
        DATA_DIR,
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        MODELS_DIR,
        ARTIFACTS_DIR,
        REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
