"""Tests for model training, evaluation and MLflow logging.

Training tests run on a small synthetic split so they stay fast. The MLflow
tests point MLflow at a temporary local directory rather than a running server,
which proves the logging code path without needing the daemon.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.evaluate import (
    CLASS_NAMES,
    build_classification_report,
    compute_metrics,
    evaluate_pipeline,
    positive_probabilities,
    select_best_model,
)
from src.exceptions import EvaluationError, MLflowError, TrainingError
from src.mlflow_utils import build_params, configure_tracking, fetch_runs, log_model_run
from src.train import (
    MODEL_PARAMS,
    build_model,
    load_model,
    load_training_report,
    model_path,
    save_model,
    train_all_models,
    train_model,
)


# --------------------------------------------------------------------------
# model construction
# --------------------------------------------------------------------------
@pytest.mark.parametrize("model_name", list(config.MODEL_NAMES))
def test_build_model_returns_unfitted_pipeline(model_name):
    from sklearn.pipeline import Pipeline

    pipeline = build_model(model_name, ["V1", "V2", "V3"])

    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["preprocessor", "model"]


def test_build_model_rejects_unknown_name():
    with pytest.raises(TrainingError, match="Unknown model"):
        build_model("gradient_boosting_magic", ["V1"])


def test_logistic_regression_uses_balanced_class_weight():
    """The imbalance decision must be visible in the estimator config."""
    estimator = build_model("logistic_regression", ["V1"]).named_steps["model"]
    assert estimator.class_weight == "balanced"


def test_random_forest_uses_balanced_class_weight():
    estimator = build_model("random_forest", ["V1"]).named_steps["model"]
    assert estimator.class_weight == "balanced_subsample"


def test_random_forest_is_not_scaled_but_logistic_regression_is():
    """Trees are scale invariant, so scaling them would be wasted work."""
    def _numeric_steps(model_name: str) -> list[str]:
        preprocessor = build_model(model_name, ["V1"]).named_steps["preprocessor"]
        by_name = {name: transformer for name, transformer, _ in preprocessor.transformers}
        return [name for name, _ in by_name["num"].steps]

    assert "scaler" in _numeric_steps("logistic_regression")
    assert "imputer" in _numeric_steps("logistic_regression")
    assert "scaler" not in _numeric_steps("random_forest")
    assert "imputer" in _numeric_steps("random_forest")


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
@pytest.mark.parametrize("model_name", list(config.MODEL_NAMES))
def test_model_trains_and_predicts(model_name, tiny_split):
    X_train, y_train, X_test, _ = tiny_split

    pipeline, report = train_model(model_name, X_train, y_train)

    predictions = pipeline.predict(X_test)
    assert predictions.shape == (len(X_test),)
    assert set(np.unique(predictions)) <= {0, 1}
    assert report["fit_seconds"] > 0
    assert report["n_features_in"] == X_train.shape[1]


def test_model_learns_the_signal(tiny_split):
    """A correctly configured model should beat the 0/1 majority baseline."""
    X_train, y_train, X_test, y_test = tiny_split

    pipeline, _ = train_model("logistic_regression", X_train, y_train)
    predictions = pipeline.predict(X_test)

    from sklearn.metrics import average_precision_score

    scores = positive_probabilities(pipeline, X_test)
    prevalence = float(np.mean(y_test.to_numpy()))
    assert average_precision_score(y_test, scores) > prevalence


def test_training_on_empty_data_raises():
    empty_X = pd.DataFrame(columns=["V1", "V2"])
    empty_y = pd.Series([], dtype="int8")
    with pytest.raises(TrainingError, match="empty"):
        train_model("random_forest", empty_X, empty_y)


def test_training_on_single_class_raises(tiny_split):
    X_train, _, _, _ = tiny_split
    single = pd.Series(np.zeros(len(X_train), dtype="int8"), index=X_train.index)
    with pytest.raises(TrainingError, match="single class"):
        train_model("random_forest", X_train, single)


def test_train_all_models_persists_every_model(tmp_path, monkeypatch, tiny_split):
    X_train, y_train, _, _ = tiny_split
    models_dir = tmp_path / "models"
    interim_dir = tmp_path / "interim"
    models_dir.mkdir()
    interim_dir.mkdir()
    monkeypatch.setattr(config, "MODELS_DIR", models_dir)
    monkeypatch.setattr(config, "TRAINING_REPORT_PATH", interim_dir / "training_report.json")

    models, reports = train_all_models(X_train, y_train)

    assert set(models) == set(config.MODEL_NAMES)
    for name in config.MODEL_NAMES:
        assert model_path(name).exists()
        assert load_model(name).predict is not None
        assert reports[name]["hyperparameters"] == MODEL_PARAMS[name]
    assert config.TRAINING_REPORT_PATH.exists()


def test_load_model_without_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    with pytest.raises(TrainingError, match="train_models"):
        load_model("random_forest")


def test_load_training_report_without_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRAINING_REPORT_PATH", tmp_path / "missing.json")
    with pytest.raises(TrainingError, match="train_models"):
        load_training_report()


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def test_compute_metrics_returns_every_documented_key():
    y_true = pd.Series([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.6, 0.8, 0.4, 0.9])

    metrics = compute_metrics(y_true, y_pred, y_proba, model_name="unit")

    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc", "average_precision"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_negatives"] == 2


def test_metrics_match_manual_confusion_matrix():
    y_true = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 0, 0])
    y_proba = np.array([0.01, 0.02, 0.03, 0.55, 0.9, 0.8, 0.2, 0.1])

    metrics = compute_metrics(y_true, y_pred, y_proba)

    # precision = tp / (tp + fp) = 2 / 3
    assert metrics["precision"] == pytest.approx(2 / 3, abs=1e-9)
    # recall = tp / (tp + fn) = 2 / 4
    assert metrics["recall"] == pytest.approx(0.5, abs=1e-9)
    # accuracy = 5 / 8
    assert metrics["accuracy"] == pytest.approx(5 / 8, abs=1e-9)


def test_metrics_handle_a_model_that_predicts_no_fraud():
    """zero_division must yield real 0s, not an exception."""
    y_true = pd.Series([0, 0, 0, 1, 1])
    y_pred = np.zeros(5, dtype=int)
    y_proba = np.full(5, 0.01)

    metrics = compute_metrics(y_true, y_pred, y_proba)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0
    assert metrics["average_precision"] == pytest.approx(float(np.mean(y_true)))


def test_classification_report_mentions_both_classes():
    report = build_classification_report(
        pd.Series([0, 0, 1, 1]), np.array([0, 1, 1, 1])
    )
    for name in CLASS_NAMES:
        assert name in report
    assert "Confusion matrix" in report


# --------------------------------------------------------------------------
# evaluation + artifacts
# --------------------------------------------------------------------------
def test_evaluate_pipeline_writes_every_artifact(tmp_path, monkeypatch, tiny_split):
    X_train, y_train, X_test, y_test = tiny_split
    pipeline, _ = train_model("random_forest", X_train, y_train)

    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")

    metrics, written = evaluate_pipeline(
        pipeline, model_name="random_forest", model_label="Random Forest",
        X_test=X_test, y_test=y_test,
    )

    names = {p.name for p in written}
    assert "confusion_matrix.png" in names
    assert "roc_curve.png" in names
    assert "precision_recall_curve.png" in names
    assert "classification_report.txt" in names
    assert "predictions.csv" in names
    assert "feature_importance.png" in names, "tree models get an importance plot"
    assert all(p.exists() and p.stat().st_size > 0 for p in written)

    predictions = pd.read_csv(tmp_path / "artifacts" / "random_forest" / "predictions.csv")
    assert {"y_true", "y_pred", "fraud_probability", "is_correct"} <= set(predictions.columns)
    assert (predictions["fraud_probability"].between(0, 1)).all()


def test_evaluate_pipeline_skips_importance_for_linear_models(tmp_path, monkeypatch, tiny_split):
    X_train, y_train, X_test, y_test = tiny_split
    pipeline, _ = train_model("logistic_regression", X_train, y_train)

    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")
    _, written = evaluate_pipeline(
        pipeline, model_name="logistic_regression", model_label="Logistic Regression",
        X_test=X_test, y_test=y_test,
    )

    assert "feature_importance.png" not in {p.name for p in written}


def test_evaluate_pipeline_on_empty_test_set_raises(tiny_split):
    X_train, y_train, X_test, y_test = tiny_split
    pipeline, _ = train_model("logistic_regression", X_train, y_train)
    with pytest.raises(EvaluationError, match="empty"):
        evaluate_pipeline(
            pipeline, model_name="logistic_regression", model_label="LR",
            X_test=X_test.iloc[0:0], y_test=y_test.iloc[0:0],
        )


# --------------------------------------------------------------------------
# model selection
# --------------------------------------------------------------------------
def test_select_best_model_uses_average_precision_first():
    results = {
        "a": {"average_precision": 0.70, "f1_score": 0.90},
        "b": {"average_precision": 0.80, "f1_score": 0.10},
    }
    best, rationale = select_best_model(results)

    assert best == "b", "average_precision is the documented primary criterion"
    assert "average_precision" in rationale


def test_select_best_model_tie_breaks_on_f1():
    results = {
        "a": {"average_precision": 0.75, "f1_score": 0.60},
        "b": {"average_precision": 0.75, "f1_score": 0.70},
    }
    best, _ = select_best_model(results)
    assert best == "b"


def test_select_best_model_with_no_results():
    best, rationale = select_best_model({})
    assert best is None
    assert rationale


# --------------------------------------------------------------------------
# MLflow
# --------------------------------------------------------------------------
@pytest.fixture
def local_mlflow(tmp_path, monkeypatch):
    """Point MLflow at a throwaway SQLite store + artifact dir - no server.

    MLflow 3 puts the filesystem FileStore in maintenance mode, so a local test
    store must use a database backend, exactly like the real project does. The
    experiment's artifact location is pinned to tmp_path so a test run never
    writes into the repository.
    """
    import mlflow

    artifact_dir = tmp_path / "mlflow_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(config, "MLFLOW_EXPERIMENT_NAME", "unit-test-experiment")

    mlflow.set_tracking_uri(uri)
    mlflow.tracking.MlflowClient().create_experiment(
        "unit-test-experiment", artifact_location=artifact_dir.as_uri()
    )
    return uri


def test_build_params_only_includes_real_hyperparameters():
    params = build_params(
        model_name="logistic_regression",
        estimator_params=MODEL_PARAMS["logistic_regression"],
        split_summary={"test_size": 0.2, "stratify": True, "feature_columns": ["V1", "V2"]},
    )

    assert params["model_name"] == "logistic_regression"
    assert params["random_state"] == config.RANDOM_STATE
    assert params["test_size"] == 0.2
    assert params["class_weight"] == "balanced"
    assert params["max_iter"] == config.LOGISTIC_REGRESSION_PARAMS["max_iter"]
    # n_estimators belongs to the forest only
    assert "n_estimators" not in params
    assert "drop_time_column" in params


def test_configure_tracking_creates_the_experiment(local_mlflow):
    _, name = configure_tracking()
    assert name == "unit-test-experiment"

    runs = fetch_runs()
    assert runs == []


def test_log_model_run_records_params_metrics_artifacts_and_model(local_mlflow, tiny_split):
    X_train, y_train, X_test, y_test = tiny_split
    pipeline, _ = train_model("logistic_regression", X_train, y_train)

    metrics, artifacts = evaluate_pipeline(
        pipeline, model_name="logistic_regression", model_label="Logistic Regression",
        X_test=X_test, y_test=y_test,
    )
    params = build_params(
        model_name="logistic_regression",
        estimator_params=MODEL_PARAMS["logistic_regression"],
        split_summary={"test_size": 0.2, "stratify": True, "feature_columns": list(X_test.columns)},
    )

    run_id = log_model_run(
        model_name="logistic_regression",
        pipeline=pipeline,
        params=params,
        metrics=metrics,
        artifact_paths=artifacts,
        X_test=X_test,
        y_true=y_test,
    )
    assert run_id

    runs = fetch_runs()
    assert len(runs) == 1
    run = runs[0]

    assert run["status"] == "FINISHED"
    assert run["param.model_name"] == "logistic_regression"
    assert run["param.class_weight"] == "balanced"
    assert run["param.drop_time_column"] == "true"
    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc", "average_precision"):
        assert f"metric.{key}" in run, f"metric {key} must be logged"
        assert 0.0 <= run[f"metric.{key}"] <= 1.0
    assert run["n_artifacts"] >= len(artifacts)


def test_logged_model_loads_and_scores_without_manual_preprocessing(local_mlflow, tiny_split):
    """The whole point of logging the Pipeline: it must be usable as-is."""
    import mlflow
    import mlflow.sklearn

    X_train, y_train, X_test, y_test = tiny_split
    pipeline, _ = train_model("random_forest", X_train, y_train)

    metrics, artifacts = evaluate_pipeline(
        pipeline, model_name="random_forest", model_label="Random Forest",
        X_test=X_test, y_test=y_test,
    )
    run_id = log_model_run(
        model_name="random_forest", pipeline=pipeline,
        params=build_params("random_forest", MODEL_PARAMS["random_forest"], {}),
        metrics=metrics, artifact_paths=artifacts, X_test=X_test, y_true=y_test,
    )

    loaded = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    predictions = loaded.predict(X_test)

    assert predictions.shape == (len(X_test),)
    np.testing.assert_array_equal(predictions, pipeline.predict(X_test))


def test_log_model_run_rejects_missing_artifacts(local_mlflow, tiny_split, tmp_path):
    X_train, y_train, X_test, y_test = tiny_split
    pipeline, _ = train_model("logistic_regression", X_train, y_train)

    with pytest.raises(MLflowError, match="missing"):
        log_model_run(
            model_name="logistic_regression", pipeline=pipeline,
            params={}, metrics={"accuracy": 0.5},
            artifact_paths=[tmp_path / "does_not_exist.png"],
            X_test=X_test, y_true=y_test,
        )


def test_configure_tracking_gives_actionable_error_when_server_is_down(monkeypatch):
    """An unreachable server must produce a message that says how to fix it.

    ``mlflow.set_experiment`` is stubbed to raise so the test exercises the
    error path deterministically and instantly, rather than waiting on the
    MLflow HTTP client's retry/backoff policy.
    """
    import mlflow

    def _boom(_name: str):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(mlflow, "set_experiment", _boom)

    with pytest.raises(MLflowError) as excinfo:
        configure_tracking(tracking_uri="http://mlflow-server-is-down.invalid")

    message = str(excinfo.value)
    assert "mlflow server" in message, "the error must tell the user what to run"
    assert "--backend-store-uri" in message
    assert "connection refused" in message, "the underlying cause must be preserved"


def test_run_summary_json_is_written(local_mlflow, tiny_split):
    X_train, y_train, X_test, y_test = tiny_split
    pipeline, _ = train_model("logistic_regression", X_train, y_train)
    metrics, artifacts = evaluate_pipeline(
        pipeline, model_name="logistic_regression", model_label="LR",
        X_test=X_test, y_test=y_test,
    )
    run_id = log_model_run(
        model_name="logistic_regression", pipeline=pipeline, params={},
        metrics=metrics, artifact_paths=artifacts, X_test=X_test, y_true=y_test,
    )

    import mlflow

    client = mlflow.tracking.MlflowClient()
    local_path = client.download_artifacts(run_id, "run_summary.json")
    payload = json.loads(open(local_path, encoding="utf-8").read())

    assert payload["model_name"] == "logistic_regression"
    assert payload["n_test_rows"] == len(X_test)
    assert "accuracy" in payload["metrics"]
