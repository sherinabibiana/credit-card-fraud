"""Shared pytest fixtures.

Tests run on small synthetic frames that mirror the real schema, so the suite
is fast, deterministic and needs neither the 151 MB dataset nor a running
MLflow server. The two integration tests that genuinely need those resources
are marked and skip themselves when the resource is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config

# Enough rows to clear config.MIN_ROWS after de-duplication.
N_ROWS = 2_000
FRAUD_RATE = 0.02


def make_frame(
    n_rows: int = N_ROWS,
    fraud_rate: float = FRAUD_RATE,
    seed: int = 0,
    with_duplicates: bool = False,
) -> pd.DataFrame:
    """Build a synthetic frame with the real dataset's schema.

    ``V*`` columns carry a weak fraud signal so a model can actually learn
    something in a test.
    """
    rng = np.random.default_rng(seed)
    n_fraud = max(int(n_rows * fraud_rate), 2)
    n_legit = n_rows - n_fraud

    def _block(count: int, label: int) -> pd.DataFrame:
        block = pd.DataFrame(
            rng.normal(loc=0.0 if label == 0 else 1.5, scale=1.0, size=(count, 28)),
            columns=[f"V{i}" for i in range(1, 29)],
        )
        block[config.TIME_COLUMN] = rng.integers(0, 172_800, size=count)
        block[config.AMOUNT_COLUMN] = rng.gamma(2.0, 40.0, size=count)
        block[config.TARGET_COLUMN] = label
        return block

    frame = pd.concat([_block(n_legit, 0), _block(n_fraud, 1)], ignore_index=True)
    frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if with_duplicates:
        repeats = frame.sample(n=max(len(frame) // 20, 1), random_state=seed)
        frame = pd.concat([frame, repeats], ignore_index=True).reset_index(drop=True)
    return frame


@pytest.fixture
def valid_frame() -> pd.DataFrame:
    """A clean, valid dataset."""
    return make_frame()


@pytest.fixture
def frame_with_duplicates() -> pd.DataFrame:
    """A valid dataset that also contains duplicate rows."""
    return make_frame(with_duplicates=True)


@pytest.fixture
def tiny_split() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """A small stratified split, for training/evaluation tests."""
    from src.preprocessing import prepare_features, split_data

    X, y, _, dropped = prepare_features(make_frame(), remove_duplicates=False)
    X_train, y_train, X_test, y_test, _ = split_data(
        X, y, n_rows_before_dedup=len(X), dropped_columns=dropped
    )
    return X_train, y_train, X_test, y_test


@pytest.fixture(autouse=True)
def isolated_artifacts(tmp_path, monkeypatch):
    """Keep generated plots/reports out of the real project directories."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "ARTIFACTS_DIR", artifact_dir)


@pytest.fixture
def dataset_available() -> bool:
    """True when the real dataset has already been downloaded."""
    return config.CACHED_DATASET_PATH.exists() and config.CACHED_DATASET_PATH.stat().st_size > 0


@pytest.fixture
def mlflow_server_available() -> bool:
    """True when an MLflow tracking server answers on the configured URI."""
    import mlflow

    try:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        client.search_experiments(max_results=1)
        return True
    except Exception:  # noqa: BLE001 - any failure means "not available"
        return False
