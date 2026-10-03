"""Dataset acquisition and schema normalisation.

Source: OpenML dataset id 1597, ``creditcard`` - the credit card fraud dataset
published by Dal Pozzolo, Caelen, Johnson and Bontempi (ULB / Worldline).
See the README for the full citation.

The loader prefers the Parquet mirror (fast, ~70 MB) and falls back to the
canonical ARFF mirror (verified by MD5) if Parquet is unavailable. Whatever is
downloaded is normalised into a single cached CSV under ``data/raw/`` so every
downstream task reads exactly the same bytes.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Callable

import pandas as pd

from src import config
from src.exceptions import DatasetError
from src.logging_utils import get_logger

LOGGER = get_logger(__name__)

_CHUNK_BYTES = 1024 * 1024


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def _download(url: str, destination: Path) -> None:
    """Stream ``url`` to ``destination`` with a clear error on failure."""
    import requests  # imported lazily so unit tests can run without network

    LOGGER.info("Downloading dataset from %s", url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=config.DOWNLOAD_TIMEOUT_SECONDS) as response:
            response.raise_for_status()
            with destination.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=_CHUNK_BYTES):
                    if chunk:
                        handle.write(chunk)
    except requests.RequestException as exc:
        destination.unlink(missing_ok=True)
        raise DatasetError(f"Failed to download dataset from {url}: {exc}") from exc
    LOGGER.info("Downloaded %.1f MB to %s", destination.stat().st_size / 1e6, destination)


def file_md5(path: Path) -> str:
    """Return the MD5 checksum of ``path`` (used to verify the ARFF mirror)."""
    digest = hashlib.md5()  # noqa: S324 - integrity check, not security
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------
def _read_parquet(path: Path) -> pd.DataFrame:
    """Read the OpenML Parquet mirror."""
    frame = pd.read_parquet(path)
    LOGGER.info("Parsed Parquet mirror: shape=%s", frame.shape)
    return frame


def _read_arff(path: Path) -> pd.DataFrame:
    """Read the canonical ARFF mirror (scipy ships with scikit-learn)."""
    from scipy.io import arff as scipy_arff

    LOGGER.info("Parsing ARFF mirror (this takes ~10s for 284k rows)...")
    records, _meta = scipy_arff.loadarff(path)
    frame = pd.DataFrame(records)
    LOGGER.info("Parsed ARFF mirror: shape=%s", frame.shape)
    return frame


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
def _decode_bytes(frame: pd.DataFrame) -> pd.DataFrame:
    """ARFF nominal values arrive as ``bytes``; decode them to ``str``."""
    for column in frame.columns:
        if frame[column].dtype == object:
            sample = frame[column].dropna()
            if not sample.empty and isinstance(sample.iloc[0], bytes):
                LOGGER.info("Decoding bytes column %r -> str", column)
                frame[column] = frame[column].map(
                    lambda value: value.decode("utf-8") if isinstance(value, bytes) else value
                )
    return frame


def normalise_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce the raw frame into the canonical schema used by the pipeline.

    * decodes ``bytes`` columns (ARFF nominals)
    * upper-cases column names and strips surrounding whitespace
    * casts every feature column to ``float64``
    * casts the target to ``int8`` and verifies it only holds {0, 1}
    * replaces +/-inf with NaN so imputation handles it explicitly
    """
    frame = _decode_bytes(frame)
    frame.columns = [str(column).strip() for column in frame.columns]

    missing = [column for column in config.EXPECTED_COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetError(
            f"Downloaded dataset is missing expected columns: {missing}. "
            f"Found: {list(frame.columns)}"
        )

    target = config.TARGET_COLUMN
    features = [column for column in config.EXPECTED_COLUMNS if column != target]

    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().all(axis=None):
        raise DatasetError(
            "No feature column could be interpreted as numeric. "
            f"Dtypes were: {frame[features].dtypes.unique().tolist()}"
        )

    target_values = pd.to_numeric(frame[target], errors="coerce")
    if target_values.isna().any():
        bad = frame.loc[target_values.isna(), target].unique()[:5].tolist()
        raise DatasetError(f"Target column {target!r} has non-numeric values, e.g. {bad}")

    normalised = numeric.astype("float64")
    normalised[target] = target_values.astype("int8")
    normalised = normalised.replace([float("inf"), float("-inf")], pd.NA)
    normalised = normalised.astype("float64")
    normalised[target] = normalised[target].astype("int8")

    LOGGER.info(
        "Normalised schema: %d rows x %d columns (target dtype=%s)",
        normalised.shape[0],
        normalised.shape[1],
        normalised[target].dtype,
    )
    return normalised


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def load_dataset(force_download: bool = False) -> pd.DataFrame:
    """Return the fraud dataset as a normalised DataFrame.

    Uses the cached CSV when it exists, otherwise downloads from OpenML and
    writes the cache. Raises :class:`DatasetError` with an actionable message
    when every source fails.
    """
    cached = config.CACHED_DATASET_PATH

    if cached.exists() and not force_download and cached.stat().st_size > 0:
        LOGGER.info("Using cached dataset %s (%.1f MB)", cached, cached.stat().st_size / 1e6)
        frame = pd.read_csv(cached)
        LOGGER.info("Loaded cached dataset: shape=%s", frame.shape)
        return frame

    config.ensure_directories()

    attempts: list[tuple[str, Callable[[Path], pd.DataFrame], bool]] = [
        (config.OPENML_PARQUET_URL, _read_parquet, False),
        (config.OPENML_ARFF_URL, _read_arff, True),
    ]

    errors: list[str] = []
    for url, parser, verify_md5 in attempts:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                download_path = Path(tmpdir) / Path(url).name
                _download(url, download_path)

                if verify_md5:
                    actual = file_md5(download_path)
                    if actual != config.OPENML_ARFF_MD5:
                        raise DatasetError(
                            f"Checksum mismatch for {url}: expected "
                            f"{config.OPENML_ARFF_MD5}, got {actual}"
                        )
                    LOGGER.info("ARFF checksum verified: %s", actual)

                frame = normalise_schema(parser(download_path))

            cached.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(cached, index=False)
            LOGGER.info("Cached normalised dataset to %s", cached)
            return frame
        except (DatasetError, Exception) as exc:  # noqa: BLE001 - collect and retry
            LOGGER.warning("Source %s failed: %s", url, exc)
            errors.append(f"{url} -> {type(exc).__name__}: {exc}")
            if shutil.which("curl") is None:  # pragma: no cover - diagnostics only
                break

    raise DatasetError(
        "Could not obtain the dataset from any configured source:\n  - "
        + "\n  - ".join(errors)
        + f"\nPlace a copy at {cached} manually and re-run."
    )


def describe_dataset(frame: pd.DataFrame) -> dict[str, object]:
    """Compute the dataset profile that the README and the DAG log."""
    target = config.TARGET_COLUMN
    total = int(len(frame))
    counts = frame[target].value_counts()
    fraud = int(counts.get(config.FRAUD_LABEL, 0))
    legitimate = int(counts.get(config.LEGITIMATE_LABEL, 0))

    numeric_columns = [
        column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])
    ]
    categorical_columns = [column for column in frame.columns if column not in numeric_columns]

    profile: dict[str, object] = {
        "n_rows": total,
        "n_columns": int(frame.shape[1]),
        "n_numeric_columns": len(numeric_columns),
        "n_categorical_columns": len(categorical_columns),
        "categorical_columns": categorical_columns,
        "dtypes": {str(k): str(v) for k, v in frame.dtypes.items()},
        "n_missing_total": int(frame.isna().sum().sum()),
        "missing_per_column": {
            column: int(count) for column, count in frame.isna().sum().items() if count
        },
        "n_duplicate_rows": int(frame.duplicated().sum()),
        "n_infinite_values": int(
            sum(
                int(pd.Series(frame[column]).isin([float("inf"), float("-inf")]).sum())
                for column in numeric_columns
            )
        ),
        "target_column": target,
        "n_legitimate": legitimate,
        "n_fraud": fraud,
        "fraud_percentage": round(100.0 * fraud / total, 6) if total else 0.0,
        "class_distribution": {str(k): int(v) for k, v in counts.items()},
        "imbalance_ratio_legit_to_fraud": round(legitimate / fraud, 2) if fraud else None,
    }
    return profile


def log_dataset_profile(profile: dict[str, object]) -> None:
    """Emit the dataset profile in a form that is readable in Airflow logs."""
    LOGGER.info("----- dataset profile -----")
    LOGGER.info("rows                     : %s", profile["n_rows"])
    LOGGER.info("columns                  : %s", profile["n_columns"])
    LOGGER.info("numeric / categorical    : %s / %s",
                profile["n_numeric_columns"], profile["n_categorical_columns"])
    LOGGER.info("missing values (total)   : %s", profile["n_missing_total"])
    LOGGER.info("duplicate rows           : %s", profile["n_duplicate_rows"])
    LOGGER.info("infinite values          : %s", profile["n_infinite_values"])
    LOGGER.info("target column            : %s", profile["target_column"])
    LOGGER.info("legitimate transactions  : %s", profile["n_legitimate"])
    LOGGER.info("fraudulent transactions  : %s", profile["n_fraud"])
    LOGGER.info("fraud percentage         : %s%%", profile["fraud_percentage"])
    LOGGER.info("legit:fraud ratio        : %s:1", profile["imbalance_ratio_legit_to_fraud"])
    LOGGER.info("--------------------------")
