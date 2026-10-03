"""Step 3/4 - feature preparation and the stratified split.

Two leakage controls are implemented here and are worth understanding:

1. **De-duplication before the split.** The source dataset contains duplicate
   rows. If those rows are split across train and test, the model is
   effectively scored on rows it memorised, so metrics look better than they
   are. De-duplication therefore happens *before* ``train_test_split``, and it
   is done on **exactly the columns the model will see** (feature columns plus
   the target). This matters: the raw file has 1,081 rows that are byte-wise
   identical, but once ``Time`` is dropped there are 8,063 rows that are
   identical *as model input* because they differed only in ``Time``. Keying
   the de-duplication on the model input is what actually closes the leak.

2. **No global preprocessing.** Imputation and scaling live inside a
   scikit-learn ``Pipeline`` that is fitted on the training data only and then
   applied to the test data. Nothing is fitted on the full dataset, so test
   statistics never influence training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.exceptions import PreprocessingError
from src.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class SplitSummary:
    """Everything needed to reproduce and audit the split."""

    test_size: float
    random_state: int
    stratify: bool
    n_rows_before_dedup: int
    n_duplicate_rows_removed: int
    n_rows_after_dedup: int
    n_train: int
    n_test: int
    train_fraud: int
    train_legitimate: int
    test_fraud: int
    test_legitimate: int
    train_fraud_pct: float
    test_fraud_pct: float
    feature_columns: list[str]
    dropped_columns: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# --------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------
def select_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return ``(feature_columns, dropped_columns)`` for this project.

    ``Time`` is dropped by default: it is the number of seconds since the first
    row of the file, i.e. a transaction-*order* index over a single two-day
    window rather than a property of the transaction itself. Keeping it lets a
    model memorise "which second of these two days was fraud-prone", a pattern
    that will not carry over to new data. Set ``FRAUD_DROP_TIME=false`` to keep
    it and compare the two configurations.
    """
    target = config.TARGET_COLUMN
    candidates = [column for column in config.EXPECTED_COLUMNS if column != target]
    candidates = [column for column in candidates if column in frame.columns]

    dropped: list[str] = []
    if config.DROP_TIME and config.TIME_COLUMN in candidates:
        candidates.remove(config.TIME_COLUMN)
        dropped.append(config.TIME_COLUMN)

    if not candidates:
        raise PreprocessingError("No feature columns left after dropping columns.")

    if dropped:
        LOGGER.info("Dropped column(s) %s (see README: 'Feature decisions')", dropped)
    LOGGER.info("Using %d feature columns: %s ...", len(candidates), candidates[:5])
    return candidates, dropped


def prepare_features(
    frame: pd.DataFrame, remove_duplicates: bool = True
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Clean the raw dataset and split it into ``X`` and ``y``.

    De-duplication is keyed on the feature columns plus the target, i.e. on
    exactly what the model will be shown, so no identical *model input* can
    appear in both partitions.

    Returns ``(X, y, feature_columns, dropped_columns)``.
    """
    if frame is None or frame.empty:
        raise PreprocessingError("Cannot prepare features from an empty dataset.")
    if config.TARGET_COLUMN not in frame.columns:
        raise PreprocessingError(
            f"Target column {config.TARGET_COLUMN!r} not found; run validation first."
        )

    rows_before = len(frame)
    feature_columns, dropped_columns = select_feature_columns(frame)

    if remove_duplicates:
        model_input = feature_columns + [config.TARGET_COLUMN]
        conflicting = (
            frame.groupby(feature_columns, dropna=False)[config.TARGET_COLUMN]
            .nunique()
            .gt(1)
        )
        if bool(conflicting.any()):
            raise PreprocessingError(
                f"{int(conflicting.sum())} feature vector(s) carry conflicting target "
                "labels. De-duplicating would silently discard real information; "
                "investigate the source data first."
            )

        n_duplicates = int(frame.duplicated(subset=model_input).sum())
        if n_duplicates:
            frame = (
                frame.drop_duplicates(subset=model_input).reset_index(drop=True)
            )
            LOGGER.info(
                "Removed %d duplicate model input(s) (%d -> %d rows) before splitting. "
                "Keyed on the %d feature columns + target, so identical transactions "
                "cannot land in both train and test.",
                n_duplicates, rows_before, len(frame), len(feature_columns),
            )

    X = frame[feature_columns].astype("float64").copy()
    y = frame[config.TARGET_COLUMN].astype("int8").copy()

    if X.empty:
        raise PreprocessingError("Feature matrix is empty after preparation.")

    LOGGER.info(
        "Prepared features: X=%s, y=%s | fraud in y: %d (%.4f%%)",
        X.shape, y.shape, int(y.sum()), 100.0 * float(y.mean()),
    )
    return X, y, feature_columns, dropped_columns


def save_prepared(
    X: pd.DataFrame, y: pd.Series, feature_columns: list[str], dropped_columns: list[str]
) -> Path:
    """Persist the prepared (cleaned, de-duplicated) dataset for the split task."""
    config.ensure_directories()
    frame = X.copy()
    frame[config.TARGET_COLUMN] = y.to_numpy()
    frame.to_csv(config.PREPARED_DATA_PATH, index=False)

    manifest = {
        "n_rows": int(len(frame)),
        "feature_columns": list(feature_columns),
        "dropped_columns": list(dropped_columns),
        "n_fraud": int(y.sum()),
        "n_legitimate": int(len(y) - int(y.sum())),
    }
    config.INTERIM_DATA_DIR.joinpath("prepared_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    LOGGER.info("Saved prepared dataset to %s", config.PREPARED_DATA_PATH)
    return config.PREPARED_DATA_PATH


def load_prepared() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Reload the prepared dataset written by :func:`save_prepared`."""
    if not config.PREPARED_DATA_PATH.exists():
        raise PreprocessingError(
            f"Prepared dataset not found at {config.PREPARED_DATA_PATH}. "
            "Run the prepare_data step first."
        )
    frame = pd.read_csv(config.PREPARED_DATA_PATH)
    y = frame.pop(config.TARGET_COLUMN).astype("int8")
    manifest_path = config.INTERIM_DATA_DIR / "prepared_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    LOGGER.info("Loaded prepared dataset: X=%s, fraud=%d", frame.shape, int(y.sum()))
    return frame, y, manifest


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
    n_rows_before_dedup: int | None = None,
    n_duplicate_rows_removed: int = 0,
    dropped_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, SplitSummary]:
    """Reproducible, stratified train/test split.

    ``stratify=y`` keeps the 0.17% fraud rate identical in both partitions,
    which matters a great deal when the minority class has only a few hundred
    members.
    """
    if not 0.0 < test_size < 1.0:
        raise PreprocessingError(f"test_size must be in (0, 1), got {test_size}.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    y_train = y_train.astype("int8")
    y_test = y_test.astype("int8")

    summary = SplitSummary(
        test_size=test_size,
        random_state=random_state,
        stratify=True,
        n_rows_before_dedup=n_rows_before_dedup if n_rows_before_dedup is not None else len(X),
        n_duplicate_rows_removed=n_duplicate_rows_removed,
        n_rows_after_dedup=len(X),
        n_train=len(X_train),
        n_test=len(X_test),
        train_fraud=int(y_train.sum()),
        train_legitimate=int((y_train == config.LEGITIMATE_LABEL).sum()),
        test_fraud=int(y_test.sum()),
        test_legitimate=int((y_test == config.LEGITIMATE_LABEL).sum()),
        train_fraud_pct=round(100.0 * float(y_train.mean()), 6),
        test_fraud_pct=round(100.0 * float(y_test.mean()), 6),
        feature_columns=list(X.columns),
        dropped_columns=list(dropped_columns or []),
    )

    _assert_disjoint(X_train, X_test)

    LOGGER.info(
        "Split done -> train %d rows (fraud %d = %.4f%%) | test %d rows "
        "(fraud %d = %.4f%%) | random_state=%d stratify=True",
        summary.n_train, summary.train_fraud, summary.train_fraud_pct,
        summary.n_test, summary.test_fraud, summary.test_fraud_pct, random_state,
    )
    return X_train, y_train, X_test, y_test, summary


def _assert_disjoint(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    """Fail loudly if any identical feature row appears in both partitions."""
    train_keys = pd.util.hash_pandas_object(X_train, index=False)
    test_keys = pd.util.hash_pandas_object(X_test, index=False)
    overlap = np.intersect1d(train_keys.to_numpy(), test_keys.to_numpy())
    if overlap.size:
        raise PreprocessingError(
            f"{overlap.size} identical row(s) appear in both train and test. "
            "This is data leakage - remove duplicates before splitting."
        )
    LOGGER.info("Leakage check passed: train and test share 0 identical rows.")


# --------------------------------------------------------------------------
# Preprocessing pipeline
# --------------------------------------------------------------------------
def build_preprocessor(
    feature_columns: list[str], scale: bool
) -> ColumnTransformer:
    """Build the preprocessing transformer.

    ``scale`` is False for tree ensembles: a random forest splits on value
    order, so standardisation cannot change its predictions and would only add
    a fitting step.
    """
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy=config.IMPUTE_STRATEGY, keep_empty_features=True))
    ]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    LOGGER.info(
        "Preprocessor: imputer(%s)%s",
        config.IMPUTE_STRATEGY, " + StandardScaler" if scale else " (no scaling)",
    )
    return ColumnTransformer(
        transformers=[("num", Pipeline(numeric_steps), feature_columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def save_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    summary: SplitSummary,
) -> Path:
    """Persist train/test partitions and the split manifest for the next task."""
    config.ensure_directories()

    def _write(features: pd.DataFrame, target: pd.Series, path: Path) -> None:
        out = features.copy()
        out[config.TARGET_COLUMN] = target.to_numpy()
        out.to_csv(path, index=False)

    _write(X_train, y_train, config.TRAIN_DATA_PATH)
    _write(X_test, y_test, config.TEST_DATA_PATH)

    config.SPLIT_MANIFEST_PATH.write_text(
        json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
    )
    LOGGER.info("Saved train/test partitions and manifest to %s", config.INTERIM_DATA_DIR)
    return config.SPLIT_MANIFEST_PATH


def load_split() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict]:
    """Reload the partitions written by :func:`save_split`."""
    if not config.TRAIN_DATA_PATH.exists() or not config.TEST_DATA_PATH.exists():
        raise PreprocessingError(
            f"Train/test files not found in {config.INTERIM_DATA_DIR}. "
            "Run the split_data step first."
        )
    train = pd.read_csv(config.TRAIN_DATA_PATH)
    test = pd.read_csv(config.TEST_DATA_PATH)
    manifest = json.loads(config.SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))

    y_train = train.pop(config.TARGET_COLUMN).astype("int8")
    y_test = test.pop(config.TARGET_COLUMN).astype("int8")
    LOGGER.info(
        "Loaded split from disk: train %s (fraud %d) | test %s (fraud %d)",
        train.shape, int(y_train.sum()), test.shape, int(y_test.sum()),
    )
    return train, y_train, test, y_test, manifest
