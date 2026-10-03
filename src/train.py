"""Step 5 - model construction and training.

Two baselines are trained on the same split:

* **Logistic Regression** - the interpretable baseline. Needs standardised
  features and a large ``max_iter`` because the classes are so far apart.
* **Random Forest** - a tree ensemble that captures non-linear interactions
  between the PCA components. Invariant to feature scale.

Both use ``class_weight`` to compensate for the 0.17% fraud rate. This is a
deliberate choice over SMOTE-style resampling: reweighting keeps the test set
untouched, adds no dependency, and cannot create synthetic fraud transactions
that blur the class boundary. The trade-off is documented in the README.

Each estimator is wrapped together with its preprocessing in a single
``Pipeline`` so that the object logged to MLflow can be loaded and used for
scoring without rebuilding any preprocessing step by hand.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.exceptions import TrainingError
from src.logging_utils import get_logger
from src.preprocessing import build_preprocessor

LOGGER = get_logger(__name__)

# Hyperparameters per model. Only keys accepted by the estimator are passed on.
MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "logistic_regression": dict(config.LOGISTIC_REGRESSION_PARAMS),
    "random_forest": dict(config.RANDOM_FOREST_PARAMS),
}

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}


def build_model(model_name: str, feature_columns: list[str]) -> Pipeline:
    """Return an unfitted preprocessing + estimator pipeline."""
    if model_name not in MODEL_PARAMS:
        raise TrainingError(
            f"Unknown model {model_name!r}. Available: {sorted(MODEL_PARAMS)}"
        )

    params = MODEL_PARAMS[model_name]
    scale = model_name in config.SCALE_FOR_MODELS

    if model_name == "logistic_regression":
        estimator = LogisticRegression(**params)
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(**params)
    else:  # pragma: no cover - guarded by the membership check above
        raise TrainingError(f"No estimator registered for {model_name!r}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_columns, scale=scale)),
            ("model", estimator),
        ]
    )
    LOGGER.info("Built pipeline %r with steps %s", model_name, [n for n, _ in pipeline.steps])
    return pipeline


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit one pipeline and return it with a small training report."""
    if X_train.empty:
        raise TrainingError(f"Cannot train {model_name!r} on an empty training set.")
    if len(set(y_train.unique())) < 2:
        raise TrainingError(
            f"Training data for {model_name!r} contains a single class; "
            "cannot fit a binary classifier."
        )

    pipeline = build_model(model_name, list(X_train.columns))
    LOGGER.info(
        "Training %s on %d rows (%d fraud)...", MODEL_LABELS[model_name],
        len(X_train), int(y_train.sum()),
    )

    started = time.perf_counter()
    try:
        pipeline.fit(X_train, y_train)
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise TrainingError(f"Training {model_name!r} failed: {exc}") from exc
    fit_seconds = time.perf_counter() - started

    estimator = pipeline.named_steps["model"]
    report = {
        "model_name": model_name,
        "model_label": MODEL_LABELS[model_name],
        "fit_seconds": round(fit_seconds, 3),
        "n_train_rows": int(len(X_train)),
        "n_train_fraud": int(y_train.sum()),
        "n_features_in": int(getattr(estimator, "n_features_in_", X_train.shape[1])),
        "hyperparameters": {
            key: value for key, value in MODEL_PARAMS[model_name].items()
        },
    }
    LOGGER.info(
        "Trained %s in %.2fs (n_features_in=%d)",
        MODEL_LABELS[model_name], fit_seconds, report["n_features_in"],
    )
    return pipeline, report


def train_all_models(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[dict[str, Pipeline], dict[str, dict[str, Any]]]:
    """Train every configured model and persist each fitted pipeline."""
    models: dict[str, Pipeline] = {}
    reports: dict[str, dict[str, Any]] = {}

    for model_name in config.MODEL_NAMES:
        pipeline, report = train_model(model_name, X_train, y_train)
        models[model_name] = pipeline
        reports[model_name] = report
        save_model(pipeline, model_name)

    save_training_report(reports)
    return models, reports


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def model_path(model_name: str) -> Path:
    return config.MODELS_DIR / f"{model_name}.joblib"


def save_model(pipeline: Pipeline, model_name: str) -> Path:
    """Persist a fitted pipeline under ``models/``."""
    config.ensure_directories()
    destination = model_path(model_name)
    joblib.dump(pipeline, destination)
    LOGGER.info("Saved fitted pipeline to %s", destination)
    return destination


def load_model(model_name: str) -> Pipeline:
    """Load a fitted pipeline from ``models/``."""
    path = model_path(model_name)
    if not path.exists():
        raise TrainingError(
            f"No saved model at {path}. Run the train_models step first."
        )
    LOGGER.info("Loading fitted pipeline from %s", path)
    return joblib.load(path)


def save_training_report(reports: dict[str, dict[str, Any]]) -> Path:
    """Persist the training report for the evaluate/log_results tasks."""
    config.ensure_directories()
    config.TRAINING_REPORT_PATH.write_text(
        json.dumps(reports, indent=2, default=str), encoding="utf-8"
    )
    LOGGER.info("Training report written to %s", config.TRAINING_REPORT_PATH)
    return config.TRAINING_REPORT_PATH


def load_training_report() -> dict[str, dict[str, Any]]:
    """Reload the training report written by :func:`save_training_report`."""
    if not config.TRAINING_REPORT_PATH.exists():
        raise TrainingError(
            f"Training report not found at {config.TRAINING_REPORT_PATH}. "
            "Run the train_models step first."
        )
    return json.loads(config.TRAINING_REPORT_PATH.read_text(encoding="utf-8"))
