"""Run the whole ML pipeline without Airflow.

This is the exact same sequence of steps the DAG executes, in the same order,
so it doubles as (a) a fast local smoke test and (b) readable documentation of
what the pipeline does.

Usage:
    python -m scripts.run_pipeline              # full run, logs to MLflow
    python -m scripts.run_pipeline --no-mlflow   # skip MLflow entirely
    python -m scripts.run_pipeline --force-download
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from src import config
from src.data_loader import describe_dataset, load_dataset, log_dataset_profile
from src.data_validation import enforce_valid_dataset, save_report
from src.evaluate import evaluate_pipeline, select_best_model
from src.exceptions import FraudPipelineError
from src.logging_utils import get_logger
from src.mlflow_utils import build_params, configure_tracking, log_model_run
from src.preprocessing import prepare_features, save_split, split_data
from src.train import MODEL_LABELS, load_model, train_all_models

LOGGER = get_logger(__name__)


def run(use_mlflow: bool = True, force_download: bool = False) -> dict[str, Any]:
    """Execute load -> validate -> prepare -> split -> train -> evaluate -> log."""
    started = time.perf_counter()
    config.ensure_directories()

    # ---- 1. load -------------------------------------------------------
    LOGGER.info("Loading dataset...")
    frame = load_dataset(force_download=force_download)
    profile = describe_dataset(frame)
    log_dataset_profile(profile)

    # ---- 2. validate ---------------------------------------------------
    LOGGER.info("Validating dataset...")
    report = enforce_valid_dataset(frame)
    save_report(report)

    # ---- 3. prepare ----------------------------------------------------
    LOGGER.info("Preparing features...")
    X, y, feature_columns, dropped_columns = prepare_features(frame)
    del frame

    # ---- 4. split ------------------------------------------------------
    LOGGER.info("Splitting data (stratified, reproducible)...")
    X_train, y_train, X_test, y_test, summary = split_data(
        X, y,
        n_rows_before_dedup=int(profile["n_rows"]),
        n_duplicate_rows_removed=int(profile["n_rows"]) - len(X),
        dropped_columns=dropped_columns,
    )
    save_split(X_train, y_train, X_test, y_test, summary)
    del X, y

    # ---- 5. train ------------------------------------------------------
    LOGGER.info("Training models...")
    models, training_reports = train_all_models(X_train, y_train)
    del X_train, y_train

    # ---- 6/7. evaluate + log -------------------------------------------
    if use_mlflow:
        LOGGER.info("Logging to MLflow...")
        configure_tracking()

    results: dict[str, dict[str, float]] = {}
    run_ids: dict[str, str] = {}

    for model_name in config.MODEL_NAMES:
        LOGGER.info("Evaluating %s...", MODEL_LABELS[model_name])
        pipeline = models.get(model_name) or load_model(model_name)
        metrics, artifacts = evaluate_pipeline(
            pipeline,
            model_name=model_name,
            model_label=MODEL_LABELS[model_name],
            X_test=X_test,
            y_test=y_test,
        )
        metrics["fit_seconds"] = float(training_reports[model_name]["fit_seconds"])
        results[model_name] = metrics

        if use_mlflow:
            params = build_params(
                model_name=model_name,
                estimator_params=training_reports[model_name]["hyperparameters"],
                split_summary=summary.to_dict(),
                extra={"pipeline": "credit_card_fraud_detection"},
            )
            run_ids[model_name] = log_model_run(
                model_name=model_name,
                pipeline=pipeline,
                params=params,
                metrics=metrics,
                artifact_paths=artifacts,
                X_test=X_test,
                y_true=y_test,
                extra_tags={
                    "trigger": "standalone_script",
                    "fraud_rows_in_test": str(int(y_test.sum())),
                },
            )

    best_model, rationale = select_best_model(results)
    elapsed = time.perf_counter() - started

    summary_out = {
        "elapsed_seconds": round(elapsed, 2),
        "split": summary.to_dict(),
        "metrics": results,
        "mlflow_run_ids": run_ids,
        "selected_model": best_model,
        "selection_rationale": rationale,
        "selection_criterion": (
            f"primary={config.MODEL_SELECTION_PRIMARY_METRIC}, "
            f"tiebreak={config.MODEL_SELECTION_TIEBREAK_METRIC}"
        ),
    }
    report_path = config.REPORTS_DIR / "pipeline_summary.json"
    report_path.write_text(json.dumps(summary_out, indent=2, default=str), encoding="utf-8")

    LOGGER.info("----- pipeline summary -----")
    for model_name, metrics in results.items():
        LOGGER.info(
            "%-22s precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f avg_precision=%.4f",
            model_name, metrics["precision"], metrics["recall"],
            metrics["f1_score"], metrics["roc_auc"], metrics["average_precision"],
        )
    LOGGER.info("selection criterion : %s", summary_out["selection_criterion"])
    LOGGER.info("selected model      : %s (%s)", best_model, rationale)
    LOGGER.info("mlflow run ids      : %s", json.dumps(run_ids))
    LOGGER.info("summary written to  : %s", report_path)
    LOGGER.info("total elapsed       : %.1fs", elapsed)
    LOGGER.info("Pipeline completed successfully.")
    return summary_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fraud detection ML pipeline.")
    parser.add_argument(
        "--no-mlflow", action="store_true",
        help="Train and evaluate without contacting the MLflow server.",
    )
    parser.add_argument(
        "--force-download", action="store_true",
        help="Ignore the local dataset cache and download again.",
    )
    args = parser.parse_args()

    try:
        run(use_mlflow=not args.no_mlflow, force_download=args.force_download)
    except FraudPipelineError as exc:
        LOGGER.error("Pipeline failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
