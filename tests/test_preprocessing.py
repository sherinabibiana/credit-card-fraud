"""Tests for feature preparation, the preprocessing pipeline and the split.

The leakage tests are the important ones: they assert the two guarantees this
project makes about the train/test boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src import config
from src.exceptions import PreprocessingError
from src.preprocessing import (
    build_preprocessor,
    load_prepared,
    prepare_features,
    save_prepared,
    select_feature_columns,
    split_data,
)
from tests.conftest import make_frame


# --------------------------------------------------------------------------
# feature selection
# --------------------------------------------------------------------------
def test_time_column_is_dropped_by_default():
    frame = make_frame()
    features, dropped = select_feature_columns(frame)

    assert config.TIME_COLUMN in dropped
    assert config.TIME_COLUMN not in features
    assert config.TARGET_COLUMN not in features


def test_time_column_can_be_kept_via_config(monkeypatch):
    monkeypatch.setattr(config, "DROP_TIME", False)
    features, dropped = select_feature_columns(make_frame())

    assert config.TIME_COLUMN in features
    assert dropped == []


def test_prepare_features_shapes_and_dtypes():
    frame = make_frame()
    X, y, features, dropped = prepare_features(frame, remove_duplicates=False)

    assert len(X) == len(frame)
    assert list(X.columns) == features
    assert config.TARGET_COLUMN not in X.columns
    assert all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)
    assert set(y.unique()) <= {0, 1}


def test_prepare_features_on_empty_frame_raises():
    with pytest.raises(PreprocessingError):
        prepare_features(pd.DataFrame())


def test_prepare_features_without_target_raises():
    frame = make_frame().drop(columns=[config.TARGET_COLUMN])
    with pytest.raises(PreprocessingError):
        prepare_features(frame)


# --------------------------------------------------------------------------
# de-duplication
# --------------------------------------------------------------------------
def test_duplicates_removed_on_the_model_input():
    frame = make_frame(with_duplicates=True)
    n_before = len(frame)

    X, y, features, _ = prepare_features(frame)

    assert len(X) < n_before, "duplicate model inputs should be removed"
    duplicated = X.duplicated(subset=features).sum()
    assert duplicated == 0, "no duplicate model input may survive"


def test_de_duplication_does_not_remove_unique_fraud_rows():
    frame = make_frame(with_duplicates=True)
    _, y_raw, _, _ = prepare_features(frame, remove_duplicates=False)
    _, y_clean, _, _ = prepare_features(frame, remove_duplicates=True)

    assert y_clean.sum() <= y_raw.sum()


def test_conflicting_labels_block_de_duplication():
    """A feature vector labelled both ways is a data problem, not a duplicate."""
    frame = make_frame(n_rows=1000)
    row = frame.iloc[[0]].copy()
    row[config.TARGET_COLUMN] = 1 - int(row[config.TARGET_COLUMN].iloc[0])
    frame = pd.concat([frame, row], ignore_index=True)

    with pytest.raises(PreprocessingError, match="conflicting"):
        prepare_features(frame)


# --------------------------------------------------------------------------
# split + leakage guarantees
# --------------------------------------------------------------------------
def test_split_is_stratified_and_reproducible():
    frame = make_frame()
    X, y, features, dropped = prepare_features(frame)

    first = split_data(X, y, dropped_columns=dropped)
    second = split_data(X, y, dropped_columns=dropped)

    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_series_equal(first[1], second[1])
    pd.testing.assert_frame_equal(first[2], second[2])
    assert first[4].to_dict() == second[4].to_dict()


def test_split_sizes_and_class_balance():
    frame = make_frame()
    X, y, features, dropped = prepare_features(frame)

    X_train, y_train, X_test, y_test, summary = split_data(
        X, y, test_size=0.2, dropped_columns=dropped
    )

    assert len(X_train) + len(X_test) == len(X)
    assert abs(len(X_test) / len(X) - 0.2) < 0.01
    # stratification keeps the fraud rate essentially identical on both sides
    assert abs(summary.train_fraud_pct - summary.test_fraud_pct) < 0.01
    assert summary.train_fraud + summary.test_fraud == int(y.sum())


def test_split_uses_fixed_random_state():
    frame = make_frame()
    X, y, _, dropped = prepare_features(frame)

    a = split_data(X, y, random_state=42, dropped_columns=dropped)
    b = split_data(X, y, random_state=7, dropped_columns=dropped)

    assert not a[0].index.equals(b[0].index), "a different seed must reshuffle"


def test_no_identical_rows_across_the_split():
    """Leakage guarantee 1: the same transaction never spans train and test."""
    frame = make_frame(with_duplicates=True)
    X, y, features, dropped = prepare_features(frame)

    X_train, _, X_test, _, _ = split_data(X, y, dropped_columns=dropped)

    train_hashes = set(pd.util.hash_pandas_object(X_train, index=False))
    test_hashes = set(pd.util.hash_pandas_object(X_test, index=False))
    assert not (train_hashes & test_hashes)


def test_split_rejects_invalid_test_size():
    X, y, _, dropped = prepare_features(make_frame())
    with pytest.raises(PreprocessingError, match="test_size"):
        split_data(X, y, test_size=1.5, dropped_columns=dropped)


# --------------------------------------------------------------------------
# preprocessing pipeline
# --------------------------------------------------------------------------
def test_preprocessor_fits_and_transforms():
    frame = make_frame()
    X, y, features, _ = prepare_features(frame, remove_duplicates=False)
    X_train, _, X_test, _, _ = split_data(X, y)

    for scale in (True, False):
        preprocessor = build_preprocessor(features, scale=scale)
        train_out = preprocessor.fit_transform(X_train)
        test_out = preprocessor.transform(X_test)

        assert train_out.shape == (len(X_train), len(features))
        assert test_out.shape == (len(X_test), len(features))
        assert np.isfinite(np.asarray(train_out)).all()


def test_scaling_only_applied_when_requested():
    frame = make_frame()
    X, _, features, _ = prepare_features(frame, remove_duplicates=False)

    scaled = build_preprocessor(features, scale=True).fit_transform(X)
    unscaled = build_preprocessor(features, scale=False).fit_transform(X)

    assert not np.allclose(scaled.std(axis=0), unscaled.std(axis=0))
    assert np.allclose(unscaled.std(axis=0), X.to_numpy().std(axis=0), rtol=1e-6)


def test_preprocessor_imputes_missing_values():
    frame = make_frame()
    X, _, features, _ = prepare_features(frame, remove_duplicates=False)
    X_with_holes = X.copy()
    X_with_holes.loc[0:9, features[0]] = np.nan

    out = build_preprocessor(features, scale=True).fit_transform(X_with_holes)
    assert np.isfinite(np.asarray(out)).all()


def test_imputer_is_fitted_on_training_data_only():
    """Leakage guarantee 2: no test statistic reaches the transformer."""
    frame = make_frame()
    X, y, features, dropped = prepare_features(frame, remove_duplicates=False)
    X_train, _, X_test, _, _ = split_data(X, y)

    train_only = build_preprocessor(features, scale=True).fit(X_train)

    # Shifting the test set must not change the fitted transformer at all.
    before = train_only.transform(X_test)
    after = train_only.transform(X_test + 1_000.0)
    assert not np.allclose(before, after)

    # Whereas refitting on everything would change it - that is the leak.
    refit = build_preprocessor(features, scale=True).fit(pd.concat([X_train, X_test]))
    assert not np.allclose(train_only.transform(X_test), refit.transform(X_test))


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------
def test_prepared_round_trip(tmp_path, monkeypatch):
    frame = make_frame()
    X, y, features, dropped = prepare_features(frame)

    monkeypatch.setattr(config, "PREPARED_DATA_PATH", tmp_path / "prepared.csv")
    monkeypatch.setattr(config, "INTERIM_DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "ensure_directories", lambda: None)

    save_prepared(X, y, features, dropped)
    X2, y2, manifest = load_prepared()

    assert list(X2.columns) == features
    pd.testing.assert_frame_equal(X, X2)
    pd.testing.assert_series_equal(y, y2)
    assert manifest["n_rows"] == len(X)


def test_load_prepared_without_file_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PREPARED_DATA_PATH", tmp_path / "missing.csv")
    with pytest.raises(PreprocessingError, match="prepare_data"):
        load_prepared()


def test_pipeline_contains_preprocessing_and_model():
    from src.train import build_model

    frame = make_frame()
    _, _, features, _ = prepare_features(frame, remove_duplicates=False)
    pipeline = build_model("logistic_regression", features)

    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["preprocessor", "model"]
