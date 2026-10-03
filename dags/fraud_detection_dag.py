"""Credit card fraud detection: Airflow orchestrates, MLflow records.

Task flow
    load_data -> validate_data -> prepare_data -> split_data
              -> train_models -> evaluate_models -> log_results

Responsibilities
    Airflow  : ordering, scheduling, retries, per-task logs in the UI
    MLflow   : one experiment run per model (params, metrics, artifacts, model)
    Sklearn  : the actual machine learning

Every task delegates the real work to the reusable ``src`` package, so the same
code path runs from the Airflow UI, from ``python -m scripts.run_pipeline`` and
from the unit tests.

Trigger it from the Airflow UI, or with:
    airflow dags trigger credit_card_fraud_detection
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pendulum
from airflow.sdk import DAG, task

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.data_loader import describe_dataset, load_dataset, log_dataset_profile  # noqa: E402
from src.data_validation import enforce_valid_dataset, save_report  # noqa: E402
from src.evaluate import evaluate_pipeline, select_best_model  # noqa: E402
from src.exceptions import FraudPipelineError  # noqa: E402
from src.logging_utils import get_logger  # noqa: E402
from src.mlflow_utils import build_params, configure_tracking, log_model_run  # noqa: E402
from src.preprocessing import (  # noqa: E402
    load_prepared,
    load_split,
    prepare_features,
    save_prepared,
    save_split,
)
from src.preprocessing import split_data as split_dataset  # noqa: E402
from src.train import MODEL_LABELS, load_model, load_training_report, train_all_models  # noqa: E402

LOGGER = get_logger("fraud_detection_dag")

# Airflow 3 TaskFlow does not unpack tuple return values, so multi-value
# hand-offs travel as a single JSON string.


@task
def load_data() -> str:
    """Locate/download the dataset and make it available downstream.

    Uses the cached CSV when present, otherwise streams it from OpenML and
    caches the normalised copy. Fails loudly if no source is reachable.
    """
    LOGGER.info("Starting load_data...")
    config.ensure_directories()

    frame = load_dataset()
    profile = describe_dataset(frame)
    log_dataset_profile(profile)

    payload = json.dumps(
        {"n_rows": profile["n_rows"], "profile": profile}, default=str
    )
    LOGGER.info("load_data finished: %d rows available.", profile["n_rows"])
    return payload


@task
def validate_data(load_payload: str) -> str:
    """Validate schema, missing values, duplicates, target and distribution.

    Raises ``DataValidationError`` on any critical problem, which fails the DAG
    run rather than training on bad data.
    """
    LOGGER.info("Starting validate_data...")
    load_info = json.loads(load_payload)

    frame = load_dataset()
    report = enforce_valid_dataset(frame)
    save_report(report)

    checks = {
        check.name: {"passed": check.passed, "severity": check.severity}
        for check in report.checks
    }
    LOGGER.info(
        "validate_data finished: %d checks, %d critical failures, %d warnings.",
        len(report.checks), len(report.critical_failures), len(report.warnings),
    )
    LOGGER.info("Critical checks passed. Row count seen: %s", load_info.get("n_rows"))
    return json.dumps({"passed": report.passed, "checks": checks})


@task
def prepare_data(validation_payload: str) -> str:
    """Clean the data, de-duplicate model inputs and separate features/target.

    Nothing is fitted here - imputation and scaling happen inside the
    scikit-learn pipeline at training time, so the test set cannot influence
    training.
    """
    LOGGER.info("Starting prepare_data...")
    validation = json.loads(validation_payload)
    if not validation.get("passed"):
        raise FraudPipelineError(
            "prepare_data refuses to run because validate_data did not pass."
        )

    frame = load_dataset()
    n_rows_raw = len(frame)
    X, y, feature_columns, dropped_columns = prepare_features(frame)
    n_removed = n_rows_raw - len(X)
    del frame

    save_prepared(X, y, feature_columns, dropped_columns)

    payload = json.dumps(
        {
            "n_rows": len(X),
            "feature_columns": feature_columns,
            "dropped_columns": dropped_columns,
            "n_fraud": int(y.sum()),
            "n_duplicate_model_inputs_removed": n_removed,
        }
    )
    LOGGER.info(
        "prepare_data finished: %d rows, %d features, dropped=%s, %d duplicate "
        "model inputs removed",
        len(X), len(feature_columns), dropped_columns, n_removed,
    )
    return payload


@task
def split_data(prepare_payload: str) -> str:
    """Reproducible stratified train/test split, persisted for later tasks.

    Note the alias: ``split_dataset`` is ``src.preprocessing.split_data``. The
    task below is *also* called ``split_data``, and the decorator rebinds that
    name in this module - calling ``split_data(...)`` inside the task body would
    hit the task object instead of the real function.
    """
    LOGGER.info("Starting split_data...")
    prepared = json.loads(prepare_payload)

    X, y, manifest = load_prepared()
    if len(X) != prepared["n_rows"]:
        raise FraudPipelineError(
            f"Prepared dataset has {len(X)} rows but prepare_data reported "
            f"{prepared['n_rows']}. Delete data/interim and re-run."
        )

    removed = int(prepared.get("n_duplicate_model_inputs_removed", 0))
    X_train, y_train, X_test, y_test, summary = split_dataset(
        X, y,
        n_rows_before_dedup=len(X) + removed,
        n_duplicate_rows_removed=removed,
        dropped_columns=manifest.get("dropped_columns", []),
    )
    save_split(X_train, y_train, X_test, y_test, summary)

    LOGGER.info(
        "split_data finished: train=%d (fraud %d) | test=%d (fraud %d)",
        summary.n_train, summary.train_fraud, summary.n_test, summary.test_fraud,
    )
    return json.dumps(summary.to_dict())


@task
def train_models(split_payload: str) -> str:
    """Fit Logistic Regression and Random Forest, persist both pipelines."""
    LOGGER.info("Starting train_models...")
    split = json.loads(split_payload)

    X_train, y_train, X_test, y_test, _ = load_split()
    if len(X_train) != split["n_train"]:
        raise FraudPipelineError(
            f"Train partition on disk has {len(X_train)} rows but the split "
            f"manifest says {split['n_train']}. Delete data/interim and re-run."
        )

    models, reports = train_all_models(X_train, y_train)
    for name, report in reports.items():
        LOGGER.info(
            "train_models: %s fitted in %.2fs on %d rows.",
            report["model_label"], report["fit_seconds"], report["n_train_rows"],
        )

    return json.dumps({name: report["model_label"] for name, report in reports.items()})


@task
def evaluate_models(train_payload: str, split_payload: str) -> str:
    """Score both models on the untouched test set and log one MLflow run each."""
    LOGGER.info("Starting evaluate_models...")
    trained = json.loads(train_payload)
    split = json.loads(split_payload)
    training_reports = load_training_report()

    # load_split() returns (X_train, y_train, X_test, y_test, manifest) - keep the
    # positions explicit, and verify against the split manifest below, because
    # scoring on the training partition still produces plausible-looking numbers.
    X_train, y_train, X_test, y_test, _ = load_split()
    if len(X_test) != split["n_test"] or int(y_test.sum()) != split["test_fraud"]:
        raise FraudPipelineError(
            f"Evaluating on the wrong partition: got {len(X_test)} rows "
            f"({int(y_test.sum())} fraud) but the split manifest expects "
            f"{split['n_test']} rows ({split['test_fraud']} fraud). "
            f"The training partition has {len(X_train)} rows "
            f"({int(y_train.sum())} fraud)."
        )

    configure_tracking()
    LOGGER.info("MLflow tracking configured at %s", config.MLFLOW_TRACKING_URI)

    results: dict[str, dict[str, float]] = {}
    run_ids: dict[str, str] = {}

    for model_name in config.MODEL_NAMES:
        model_label = MODEL_LABELS[model_name]
        LOGGER.info("Evaluating %s on %d held-out rows...", model_label, len(X_test))

        pipeline = load_model(model_name)
        metrics, artifacts = evaluate_pipeline(
            pipeline,
            model_name=model_name,
            model_label=model_label,
            X_test=X_test,
            y_test=y_test,
        )
        metrics["fit_seconds"] = float(training_reports[model_name]["fit_seconds"])
        results[model_name] = metrics

        run_ids[model_name] = log_model_run(
            model_name=model_name,
            pipeline=pipeline,
            params=build_params(
                model_name=model_name,
                estimator_params=training_reports[model_name]["hyperparameters"],
                split_summary=json.loads(split_payload),
                extra={"pipeline": "credit_card_fraud_detection"},
            ),
            metrics=metrics,
            artifact_paths=artifacts,
            X_test=X_test,
            y_true=y_test,
            extra_tags={
                "trigger": "airflow_dag",
                "dag_id": "credit_card_fraud_detection",
                "fraud_rows_in_test": str(int(y_test.sum())),
            },
        )

    best_model, rationale = select_best_model(results)
    LOGGER.info("evaluate_models finished. Selected model: %s", best_model)

    return json.dumps(
        {"metrics": results, "run_ids": run_ids,
         "selected_model": best_model, "rationale": rationale}
    )


@task
def log_results(evaluate_payload: str, split_payload: str) -> str:
    """Record the final summary for the pipeline logs and reports directory."""
    LOGGER.info("Starting log_results...")
    evaluated = json.loads(evaluate_payload)
    split = json.loads(split_payload)

    metrics = evaluated["metrics"]
    criterion = (
        f"primary={config.MODEL_SELECTION_PRIMARY_METRIC}, "
        f"tiebreak={config.MODEL_SELECTION_TIEBREAK_METRIC}"
    )

    summary = {
        "dag_id": "credit_card_fraud_detection",
        "experiment_name": config.MLFLOW_EXPERIMENT_NAME,
        "tracking_uri": config.MLFLOW_TRACKING_URI,
        "split": split,
        "metrics": metrics,
        "mlflow_run_ids": evaluated["run_ids"],
        "selection_criterion": criterion,
        "selected_model": evaluated["selected_model"],
        "selection_rationale": evaluated["rationale"],
        "note": (
            "Educational MLOps demo. These numbers describe a single stratified "
            "split of one public dataset and are not evidence of production "
            "readiness."
        ),
    }
    destination = config.REPORTS_DIR / "airflow_pipeline_summary.json"
    destination.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    width = 22
    LOGGER.info("=" * 78)
    LOGGER.info("CREDIT CARD FRAUD DETECTION - PIPELINE RESULTS")
    LOGGER.info("=" * 78)
    LOGGER.info(
        "train rows=%s (fraud %s) | test rows=%s (fraud %s)",
        split["n_train"], split["train_fraud"], split["n_test"], split["test_fraud"],
    )
    header = f"{'model':<{width}}{'precision':>11}{'recall':>9}{'f1':>9}{'roc_auc':>10}{'avg_prec':>10}"
    LOGGER.info(header)
    LOGGER.info("-" * 78)
    for name, values in metrics.items():
        LOGGER.info(
            f"{name:<{width}}{values['precision']:>11.4f}{values['recall']:>9.4f}"
            f"{values['f1_score']:>9.4f}{values['roc_auc']:>10.4f}"
            f"{values['average_precision']:>10.4f}"
        )
    LOGGER.info("-" * 78)
    LOGGER.info("selection criterion : %s", criterion)
    LOGGER.info("selected model      : %s", evaluated["selected_model"])
    LOGGER.info("rationale           : %s", evaluated["rationale"])
    LOGGER.info("mlflow experiment   : %s at %s",
                config.MLFLOW_EXPERIMENT_NAME, config.MLFLOW_TRACKING_URI)
    LOGGER.info("mlflow run ids      : %s", json.dumps(evaluated["run_ids"]))
    LOGGER.info("artifacts           : %s", config.ARTIFACTS_DIR)
    LOGGER.info("summary written to  : %s", destination)
    LOGGER.info("=" * 78)
    LOGGER.info("Pipeline completed successfully.")

    return json.dumps(
        {"selected_model": evaluated["selected_model"], "summary_path": str(destination)}
    )


with DAG(
    dag_id="credit_card_fraud_detection",
    description=(
        "Detect credit card fraud with imbalanced binary classification. "
        "Airflow orchestrates the workflow; MLflow tracks every model run."
    ),
    doc_md=__doc__,
    schedule=None,  # manual trigger only - unpause and press "Trigger DAG"
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "fraud-detection", "imbalanced", "mlflow", "classification"],
    default_args={
        "owner": "mlops",
        "retries": 1,
        "retry_delay": pendulum.duration(minutes=1),
    },
) as dag:
    loaded = load_data()
    validated = validate_data(loaded)
    prepared = prepare_data(validated)
    split = split_data(prepared)
    trained = train_models(split)
    evaluated = evaluate_models(trained, split)
    log_results(evaluated, split)
