"""MLflow integration - experiment setup and per-model run logging.

One MLflow run is created per model so the UI shows a side-by-side comparison
table. Each run carries:

* **Parameters** - only those that genuinely exist on the fitted estimator,
  plus the split/preprocessing configuration.
* **Metrics** - accuracy, precision, recall, F1, ROC-AUC, average precision and
  the raw confusion-matrix counts.
* **Artifacts** - every PNG, the classification report, the prediction dump and
  the metrics JSON.
* **Model** - the complete ``Pipeline`` (preprocessing *and* estimator) with an
  inferred signature, so it can be loaded and used for scoring directly.

If the tracking server cannot be reached the module raises
:class:`MLflowError` with an actionable message rather than silently dropping
the telemetry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from src import config
from src.exceptions import MLflowError
from src.logging_utils import get_logger

LOGGER = get_logger(__name__)

# Metrics that are meaningful floats; confusion counts are logged separately.
_NUMERIC_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "average_precision",
    "specificity",
)
_COUNT_METRICS = ("true_negatives", "false_positives", "false_negatives", "true_positives")


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------
def configure_tracking(
    tracking_uri: str | None = None, experiment_name: str | None = None
) -> tuple[str, str]:
    """Point MLflow at the tracking server and make sure the experiment exists."""
    uri = tracking_uri or config.MLFLOW_TRACKING_URI
    name = experiment_name or config.MLFLOW_EXPERIMENT_NAME

    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)

    try:
        experiment = mlflow.set_experiment(name)
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise MLflowError(
            f"Could not reach the MLflow tracking server at {uri} "
            f"(experiment {name!r}). Start it with:\n"
            f"    mlflow server --host 0.0.0.0 --port {config.MLFLOW_PORT} "
            f"--backend-store-uri sqlite:///{config.PROJECT_ROOT / 'mlflow.db'} "
            f"--default-artifact-root {config.PROJECT_ROOT / 'mlruns'}\n"
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc

    LOGGER.info(
        "MLflow tracking URI=%s experiment=%r experiment_id=%s",
        uri, experiment.name, experiment.experiment_id,
    )
    return uri, experiment.name


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
def build_params(
    model_name: str,
    estimator_params: dict[str, Any],
    split_summary: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the MLflow parameter set for one run.

    Only parameters the estimator actually accepts are logged; anything that
    does not exist on this model is skipped rather than logged as ``None``.
    """
    params: dict[str, Any] = {
        "model_name": model_name,
        "random_state": config.RANDOM_STATE,
        "test_size": split_summary.get("test_size", config.TEST_SIZE),
        "stratify": split_summary.get("stratify", True),
        "drop_time_column": config.DROP_TIME,
        "impute_strategy": config.IMPUTE_STRATEGY,
        "feature_scaling": model_name in config.SCALE_FOR_MODELS,
        "n_features": len(split_summary.get("feature_columns", [])),
        "duplicate_rows_removed": split_summary.get("n_duplicate_rows_removed", 0),
    }
    for key, value in estimator_params.items():
        if value is not None:
            params[key] = value

    for key, value in (extra or {}).items():
        if value is not None:
            params[key] = value

    LOGGER.debug("Parameters for %s: %s", model_name, params)
    return params


# --------------------------------------------------------------------------
# Run logging
# --------------------------------------------------------------------------
def log_model_run(
    model_name: str,
    pipeline,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifact_paths: list[Path],
    X_test,
    y_true: Any,
    extra_tags: dict[str, str] | None = None,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    input_example_rows: int = 5,
) -> str:
    """Create one MLflow run for a model and return its run id.

    Exceptions are deliberately allowed to propagate: a failed log must fail the
    Airflow task, and MLflow then marks the run as ``FAILED`` itself.
    """
    configure_tracking(tracking_uri, experiment_name)

    run_name = model_name.replace("_", " ").title()
    tags = {"pipeline_step": "evaluate_models", "framework": "scikit-learn"}
    tags.update(extra_tags or {})

    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        mlflow.log_params({key: _stringify(value) for key, value in params.items()})

        loggable = {
            key: float(metrics[key]) for key in _NUMERIC_METRICS if key in metrics
        }
        loggable.update({key: float(metrics[key]) for key in _COUNT_METRICS if key in metrics})
        if "fit_seconds" in metrics:
            loggable["fit_seconds"] = float(metrics["fit_seconds"])
        mlflow.log_metrics(loggable)

        missing = [
            path for path in artifact_paths
            if not Path(path).exists()
        ]
        if missing:
            raise MLflowError(
                f"Expected artifact file(s) missing before logging: "
                f"{[str(path) for path in missing]}"
            )
        for path in artifact_paths:
            mlflow.log_artifact(str(path))

        mlflow.log_text(
            json.dumps(
                {
                    "model_name": model_name,
                    "metrics": metrics,
                    "params": params,
                    "n_test_rows": int(len(X_test)),
                    "n_test_fraud": int(sum(y_true)),
                },
                indent=2,
                default=str,
            ),
            "run_summary.json",
        )

        signature = infer_signature(X_test, pipeline.predict(X_test.iloc[:input_example_rows]))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            input_example=X_test.iloc[:input_example_rows],
            skops_trusted_types=config.MLFLOW_SKOPS_TRUSTED_TYPES,
        )
        run_id = run.info.run_id

    LOGGER.info(
        "Logged MLflow run %r for %s (run_id=%s, %d artifacts)",
        run_name, model_name, run_id, len(artifact_paths),
    )
    return run_id


def _stringify(value: Any) -> str:
    """MLflow params must be scalars; render bools/nones predictably."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


# --------------------------------------------------------------------------
# Read-back helpers (used by tests and by log_results)
# --------------------------------------------------------------------------
def fetch_runs(
    tracking_uri: str | None = None, experiment_name: str | None = None
) -> list[dict[str, Any]]:
    """Return every finished run in the experiment as a flat dict.

    Useful for verifying from the command line that an Airflow-triggered run
    really landed in MLflow.
    """
    configure_tracking(tracking_uri, experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name or config.MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise MLflowError(
            f"Experiment {experiment_name or config.MLFLOW_EXPERIMENT_NAME!r} does not exist yet."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        output_format="list",
        order_by=["attributes.start_time DESC"],
    )

    client = mlflow.tracking.MlflowClient()
    flattened: list[dict[str, Any]] = []
    for run in runs:
        record: dict[str, Any] = {
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName"),
            "status": run.info.status,
            "start_time": run.info.start_time,
        }
        record.update({f"param.{k}": v for k, v in run.data.params.items()})
        record.update({f"metric.{k}": v for k, v in run.data.metrics.items()})
        record["n_artifacts"] = len(client.list_artifacts(run.info.run_id))
        flattened.append(record)
    return flattened
