"""Step 6 - evaluation on the held-out test set.

Accuracy is computed but explicitly *not* treated as the headline number: with
99.83% legitimate transactions, a model that predicts "legitimate" every time
scores 99.83% accuracy while catching no fraud at all. The metrics that matter
here are precision, recall, F1 and especially average precision (PR-AUC).

Artifacts written to ``artifacts/<model_name>/``:

* ``confusion_matrix.png``
* ``roc_curve.png``
* ``precision_recall_curve.png``
* ``classification_report.txt``
* ``predictions.csv``
* ``feature_importance.png`` (tree models only)
* ``metrics.json``
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: required inside Airflow workers

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config  # noqa: E402
from src.exceptions import EvaluationError  # noqa: E402
from src.logging_utils import get_logger  # noqa: E402

LOGGER = get_logger(__name__)

CLASS_NAMES = ["legitimate", "fraudulent"]
METRIC_KEYS = (
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "average_precision",
    "true_negatives",
    "false_positives",
    "false_negatives",
    "true_positives",
    "specificity",
)


def artifacts_dir(model_name: str) -> Path:
    """Return (and create) the artifact directory for one model."""
    path = config.ARTIFACTS_DIR / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def compute_metrics(
    y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray, model_name: str = "model"
) -> dict[str, float]:
    """Compute the full metric set for one model.

    ``zero_division=0`` keeps precision/recall/F1 defined even if a model
    predicts no fraud at all - in that case they are genuinely 0, not an error.
    """
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "average_precision": float(average_precision_score(y_true, y_proba)),
            "true_negatives": float(tn),
            "false_positives": float(fp),
            "false_negatives": float(fn),
            "true_positives": float(tp),
            "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        }
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise EvaluationError(f"Metric computation failed: {exc}") from exc

    LOGGER.info(
        "%s -> accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f "
        "average_precision=%.4f (tp=%d fp=%d fn=%d tn=%d)",
        model_name, metrics["accuracy"], metrics["precision"], metrics["recall"],
        metrics["f1_score"], metrics["roc_auc"], metrics["average_precision"],
        int(metrics["true_positives"]), int(metrics["false_positives"]),
        int(metrics["false_negatives"]), int(metrics["true_negatives"]),
    )
    return metrics


def positive_probabilities(pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return P(fraud) for each row.

    Falls back to a hard prediction when an estimator exposes no
    ``predict_proba`` (not the case for the two models used here, but it keeps
    the evaluation code honest).
    """
    if hasattr(pipeline, "predict_proba"):
        return np.asarray(pipeline.predict_proba(X))[:, 1]
    LOGGER.warning("Model has no predict_proba; falling back to 0/1 predictions.")
    return np.asarray(pipeline.predict(X), dtype="float64")


# --------------------------------------------------------------------------
# Text artifacts
# --------------------------------------------------------------------------
def build_classification_report(y_true: pd.Series, y_pred: np.ndarray) -> str:
    """Per-class precision/recall/F1/support as text."""
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    header = (
        "Classification report (positive class = 'fraudulent')\n"
        "Note: accuracy is dominated by the legitimate class; read the\n"
        "fraudulent row and the macro/weighted averages instead.\n"
    )
    confusion = confusion_matrix(y_true, y_pred, labels=[0, 1])
    footer = (
        "\nConfusion matrix (rows = actual, cols = predicted)\n"
        f"{'':>12}{'legitimate':>14}{'fraudulent':>14}\n"
        f"{'legitimate':>12}{confusion[0, 0]:>14d}{confusion[0, 1]:>14d}\n"
        f"{'fraudulent':>12}{confusion[1, 0]:>14d}{confusion[1, 1]:>14d}\n"
    )
    return f"{header}\n{report}{footer}"


# --------------------------------------------------------------------------
# Plot artifacts
# --------------------------------------------------------------------------
def _save_figure(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    LOGGER.info("Wrote %s", path)
    return path


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, model_label: str, path: Path) -> Path:
    """Annotated confusion matrix heatmap (no seaborn dependency)."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks([0, 1], labels=CLASS_NAMES)
    ax.set_yticks([0, 1], labels=CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(f"Confusion matrix - {model_label}")

    threshold = matrix.max() / 2.0 if matrix.max() else 0.5
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(
                col, row, f"{value:,}",
                ha="center", va="center",
                color="white" if value > threshold else "black",
                fontsize=13, fontweight="bold",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return _save_figure(fig, path)


def plot_roc_curve(y_true: pd.Series, y_proba: np.ndarray, roc_auc: float, model_label: str, path: Path) -> Path:
    """ROC curve against the no-skill diagonal."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot(fpr, tpr, label=f"{model_label} (ROC-AUC = {roc_auc:.4f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="random baseline")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    return _save_figure(fig, path)


def plot_precision_recall_curve(
    y_true: pd.Series, y_proba: np.ndarray, average_precision: float,
    model_label: str, path: Path,
) -> Path:
    """Precision-recall curve with the no-skill baseline at the fraud rate."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    prevalence = float(np.mean(np.asarray(y_true)))
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.step(recall, precision, where="post", label=f"{model_label} (AP = {average_precision:.4f})", linewidth=2)
    ax.axhline(prevalence, color="k", linestyle="--", linewidth=1, alpha=0.7,
               label=f"no-skill baseline ({prevalence:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.25)
    return _save_figure(fig, path)


def plot_feature_importance(
    pipeline, feature_columns: list[str], model_label: str, path: Path, top_n: int = 15
) -> Path | None:
    """Top feature importances for tree models. Returns None for linear models."""
    estimator = pipeline.named_steps.get("model")
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        LOGGER.info("No feature_importances_ on %s; skipping importance plot.", model_label)
        return None

    series = pd.Series(np.asarray(importances), index=feature_columns).sort_values(ascending=True)
    shown = series.tail(top_n)
    fig, ax = plt.subplots(figsize=(6.4, max(3.2, 0.32 * len(shown) + 1.4)))
    ax.barh(shown.index, shown.to_numpy(), color="#4C72B0")
    ax.set_xlabel("Gini importance")
    ax.set_title(f"Top {len(shown)} features - {model_label}")
    ax.grid(axis="x", alpha=0.25)
    return _save_figure(fig, path)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def evaluate_pipeline(
    pipeline,
    model_name: str,
    model_label: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_artifacts: bool = True,
) -> tuple[dict[str, float], list[Path]]:
    """Score one fitted pipeline and (optionally) write all artifact files."""
    if X_test.empty:
        raise EvaluationError(f"Test set for {model_name!r} is empty.")

    y_pred = np.asarray(pipeline.predict(X_test))
    y_proba = positive_probabilities(pipeline, X_test)
    metrics = compute_metrics(y_test, y_pred, y_proba, model_name=model_name)

    written: list[Path] = []
    if not save_artifacts:
        return metrics, written

    directory = artifacts_dir(model_name)

    written.append(directory / "classification_report.txt")
    written[-1].write_text(build_classification_report(y_test, y_pred), encoding="utf-8")

    written.append(plot_confusion_matrix(y_test, y_pred, model_label, directory / "confusion_matrix.png"))
    written.append(
        plot_roc_curve(y_test, y_proba, metrics["roc_auc"], model_label, directory / "roc_curve.png")
    )
    written.append(
        plot_precision_recall_curve(
            y_test, y_proba, metrics["average_precision"], model_label,
            directory / "precision_recall_curve.png",
        )
    )

    importance_path = plot_feature_importance(
        pipeline, list(X_test.columns), model_label, directory / "feature_importance.png"
    )
    if importance_path is not None:
        written.append(importance_path)

    predictions = X_test.copy()
    predictions["y_true"] = y_test.to_numpy()
    predictions["y_pred"] = y_pred
    predictions["fraud_probability"] = y_proba
    predictions["is_correct"] = predictions["y_true"] == predictions["y_pred"]
    if len(predictions) > config.PREDICTION_DUMP_LIMIT:
        LOGGER.info(
            "Dumping only the %d highest-risk test rows to predictions.csv "
            "(full test set has %d rows).",
            config.PREDICTION_DUMP_LIMIT, len(predictions),
        )
        predictions = predictions.nlargest(config.PREDICTION_DUMP_LIMIT, "fraud_probability")
    predictions_path = directory / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    written.append(predictions_path)

    metrics_path = directory / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    written.append(metrics_path)

    LOGGER.info("Wrote %d artifact file(s) for %s", len(written), model_label)
    return metrics, written


def select_best_model(results: dict[str, dict[str, float]]) -> tuple[str | None, str]:
    """Pick a model using an explicit, documented criterion.

    Primary metric: ``average_precision`` (PR-AUC) - under 0.2% prevalence it
    is far more informative than ROC-AUC. Tie-break: ``f1_score``. The
    criterion lives in :mod:`src.config` rather than being buried here.
    """
    if not results:
        return None, "no models were evaluated"

    primary = config.MODEL_SELECTION_PRIMARY_METRIC
    tiebreak = config.MODEL_SELECTION_TIEBREAK_METRIC
    best = max(results, key=lambda name: (results[name][primary], results[name][tiebreak]))
    explanation = (
        f"selected {best!r} by highest {primary} "
        f"({results[best][primary]:.4f}), tie-break on {tiebreak} "
        f"({results[best][tiebreak]:.4f})"
    )
    LOGGER.info("Model selection: %s", explanation)
    return best, explanation
