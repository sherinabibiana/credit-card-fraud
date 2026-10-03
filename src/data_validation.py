"""Step 2 - data validation.

Runs every quality gate before a single row is used for training. All checks
are collected first and reported together, so a broken dataset produces one
actionable error instead of a chain of confusing ones.

Critical failures raise :class:`DataValidationError`, which fails the Airflow
task. Nothing is silently repaired or dropped here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.exceptions import DataValidationError
from src.logging_utils import get_logger

LOGGER = get_logger(__name__)

SEVERITY_CRITICAL = "critical"
SEVERITY_WARNING = "warning"


@dataclass
class CheckResult:
    """Outcome of a single validation check."""

    name: str
    passed: bool
    severity: str
    message: str
    details: dict[str, object] = field(default_factory=dict)

    @property
    def is_blocking(self) -> bool:
        """True when this failure must stop the pipeline."""
        return not self.passed and self.severity == SEVERITY_CRITICAL


@dataclass
class ValidationReport:
    """Aggregated result of :func:`validate_dataset`."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def critical_failures(self) -> list[CheckResult]:
        return [check for check in self.checks if check.is_blocking]

    @property
    def warnings(self) -> list[CheckResult]:
        return [
            check for check in self.checks if not check.passed and not check.is_blocking
        ]

    @property
    def passed(self) -> bool:
        return not self.critical_failures

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "n_checks": len(self.checks),
            "n_critical_failures": len(self.critical_failures),
            "n_warnings": len(self.warnings),
            "checks": [asdict(check) for check in self.checks],
        }

    def summary(self) -> str:
        """Human-readable multi-line summary for the Airflow task log."""
        lines = [
            "----- validation report -----",
            f"checks run     : {len(self.checks)}",
            f"critical fails : {len(self.critical_failures)}",
            f"warnings       : {len(self.warnings)}",
        ]
        for check in self.checks:
            mark = "PASS" if check.passed else ("FAIL" if check.is_blocking else "WARN")
            lines.append(f"[{mark}] {check.name}: {check.message}")
        lines.append("-----------------------------")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------
def _check_present(frame: pd.DataFrame | None) -> CheckResult:
    if frame is None:
        return CheckResult(
            "dataset_present", False, SEVERITY_CRITICAL,
            "No dataset was provided. Did the load_data step run?",
        )
    return CheckResult(
        "dataset_present", True, SEVERITY_CRITICAL,
        f"Dataset received with shape {frame.shape}.",
        {"shape": list(frame.shape)},
    )


def _check_not_empty(frame: pd.DataFrame) -> CheckResult:
    n_rows = len(frame)
    if n_rows == 0:
        return CheckResult(
            "dataset_not_empty", False, SEVERITY_CRITICAL, "Dataset contains no rows."
        )
    if n_rows < config.MIN_ROWS:
        return CheckResult(
            "dataset_not_empty", False, SEVERITY_CRITICAL,
            f"Dataset has {n_rows} rows, below the minimum of {config.MIN_ROWS}. "
            "The download is probably truncated.",
            {"n_rows": n_rows, "minimum": config.MIN_ROWS},
        )
    return CheckResult(
        "dataset_not_empty", True, SEVERITY_CRITICAL,
        f"Dataset has {n_rows} rows (>= {config.MIN_ROWS}).",
        {"n_rows": n_rows},
    )


def _check_target_column(frame: pd.DataFrame) -> CheckResult:
    target = config.TARGET_COLUMN
    if target not in frame.columns:
        return CheckResult(
            "target_column_present", False, SEVERITY_CRITICAL,
            f"Target column {target!r} is missing. Columns: {list(frame.columns)[:10]}...",
        )
    return CheckResult(
        "target_column_present", True, SEVERITY_CRITICAL,
        f"Target column {target!r} is present.",
    )


def _check_required_columns(frame: pd.DataFrame) -> CheckResult:
    missing = [column for column in config.EXPECTED_COLUMNS if column not in frame.columns]
    if missing:
        return CheckResult(
            "required_columns_present", False, SEVERITY_CRITICAL,
            f"Missing {len(missing)} required column(s): {missing}",
            {"missing": missing},
        )
    unexpected = [
        column for column in frame.columns if column not in config.EXPECTED_COLUMNS
    ]
    if unexpected:
        return CheckResult(
            "required_columns_present", True, SEVERITY_WARNING,
            f"All required columns present; {len(unexpected)} unexpected extra "
            f"column(s) will be ignored: {unexpected}",
            {"unexpected": unexpected},
        )
    return CheckResult(
        "required_columns_present", True, SEVERITY_CRITICAL,
        f"All {len(config.EXPECTED_COLUMNS)} required columns are present.",
    )


def _check_missing_values(frame: pd.DataFrame) -> CheckResult:
    total_cells = frame.shape[0] * frame.shape[1]
    missing_total = int(frame.isna().sum().sum())
    ratio = missing_total / total_cells if total_cells else 0.0
    per_column = {column: int(count) for column, count in frame.isna().sum().items() if count}

    if ratio > config.MAX_MISSING_RATIO:
        return CheckResult(
            "missing_values", False, SEVERITY_CRITICAL,
            f"{missing_total} missing values ({ratio:.2%}) exceed the allowed "
            f"{config.MAX_MISSING_RATIO:.2%}.",
            {"missing_total": missing_total, "ratio": ratio, "per_column": per_column},
        )
    severity = SEVERITY_WARNING if missing_total else SEVERITY_CRITICAL
    message = (
        f"{missing_total} missing values ({ratio:.4%}) - handled by the imputer."
        if missing_total
        else "No missing values."
    )
    return CheckResult(
        "missing_values", True, severity, message,
        {"missing_total": missing_total, "ratio": ratio, "per_column": per_column},
    )


def _check_infinite_values(frame: pd.DataFrame) -> CheckResult:
    numeric = frame.select_dtypes(include=[np.number])
    infinite_mask = np.isinf(numeric.to_numpy(dtype="float64", na_value=np.nan))
    n_infinite = int(infinite_mask.sum())
    if n_infinite:
        columns = [
            column for column in numeric.columns
            if np.isinf(numeric[column].to_numpy(dtype="float64", na_value=np.nan)).any()
        ]
        return CheckResult(
            "infinite_values", True, SEVERITY_WARNING,
            f"{n_infinite} infinite value(s) found in {columns}; they are treated as "
            "missing and imputed.",
            {"n_infinite": n_infinite, "columns": columns},
        )
    return CheckResult(
        "infinite_values", True, SEVERITY_CRITICAL, "No infinite values found.",
        {"n_infinite": 0},
    )


def _check_duplicates(frame: pd.DataFrame) -> CheckResult:
    """Report duplicate rows on the raw frame and on the model input.

    Two numbers matter and they differ: byte-wise duplicates in the file, and
    duplicates of the columns the model actually consumes (some rows differ
    only in ``Time``, which this project drops).
    """
    model_input = [
        column for column in config.EXPECTED_COLUMNS
        if column in frame.columns
        and column not in (config.TARGET_COLUMN, config.TIME_COLUMN)
    ] + [config.TARGET_COLUMN]

    n_exact = int(frame.duplicated().sum())
    n_model_input = (
        int(frame.duplicated(subset=model_input).sum())
        if all(column in frame.columns for column in model_input)
        else n_exact
    )
    details = {
        "n_duplicate_rows_exact": n_exact,
        "n_duplicate_model_inputs": n_model_input,
    }

    if (n_exact or n_model_input) and not config.ALLOW_DUPLICATE_ROWS:
        return CheckResult(
            "duplicate_rows", False, SEVERITY_CRITICAL,
            f"{n_exact} duplicate row(s) ({n_model_input} duplicate model inputs) "
            "found and duplicates are not allowed.",
            details,
        )

    n_removed = n_model_input or n_exact
    return CheckResult(
        "duplicate_rows", True, SEVERITY_WARNING if n_removed else SEVERITY_CRITICAL,
        f"{n_exact} byte-wise duplicate row(s) and {n_model_input} duplicate model "
        f"input(s) found. prepare_data removes the {n_removed} repeated model "
        "inputs before the split so the same transaction cannot appear in both "
        "train and test.",
        details,
    )


def _check_target_values(frame: pd.DataFrame) -> CheckResult:
    target = config.TARGET_COLUMN
    observed = set(pd.unique(frame[target].dropna()))
    expected = {config.LEGITIMATE_LABEL, config.FRAUD_LABEL}
    unexpected = sorted(observed - expected, key=str)
    if unexpected:
        return CheckResult(
            "target_values_valid", False, SEVERITY_CRITICAL,
            f"Target column contains unexpected value(s) {unexpected}; expected "
            f"exactly {sorted(expected)} (0 = legitimate, 1 = fraud).",
            {"observed": sorted(map(str, observed)), "unexpected": unexpected},
        )
    return CheckResult(
        "target_values_valid", True, SEVERITY_CRITICAL,
        f"Target only contains the expected classes {sorted(expected)}.",
        {"observed": sorted(map(str, observed))},
    )


def _check_target_distribution(frame: pd.DataFrame) -> CheckResult:
    target = config.TARGET_COLUMN
    counts = frame[target].value_counts()
    n_fraud = int(counts.get(config.FRAUD_LABEL, 0))
    n_legitimate = int(counts.get(config.LEGITIMATE_LABEL, 0))
    total = n_fraud + n_legitimate
    fraud_pct = 100.0 * n_fraud / total if total else 0.0

    if n_fraud == 0 or n_legitimate == 0:
        return CheckResult(
            "target_distribution", False, SEVERITY_CRITICAL,
            f"Both classes must be present. Found {n_legitimate} legitimate and "
            f"{n_fraud} fraudulent rows.",
            {"n_legitimate": n_legitimate, "n_fraud": n_fraud},
        )
    return CheckResult(
        "target_distribution", True, SEVERITY_CRITICAL,
        f"{n_legitimate} legitimate ({100 - fraud_pct:.4f}%) vs {n_fraud} fraudulent "
        f"({fraud_pct:.4f}%) - ratio {n_legitimate / n_fraud:.1f}:1. Accuracy alone "
        "is meaningless here; precision/recall/PR-AUC are used instead.",
        {
            "n_legitimate": n_legitimate,
            "n_fraud": n_fraud,
            "fraud_percentage": round(fraud_pct, 6),
            "imbalance_ratio": round(n_legitimate / n_fraud, 2),
        },
    )


def _check_numeric_features(frame: pd.DataFrame) -> CheckResult:
    features = [column for column in frame.columns if column != config.TARGET_COLUMN]
    non_numeric = [
        column for column in features
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        return CheckResult(
            "features_numeric", False, SEVERITY_CRITICAL,
            f"{len(non_numeric)} feature column(s) are not numeric: {non_numeric}. "
            "This project expects the PCA-transformed numeric OpenML schema.",
            {"non_numeric": non_numeric},
        )
    return CheckResult(
        "features_numeric", True, SEVERITY_CRITICAL,
        f"All {len(features)} feature columns are numeric.",
        {"n_features": len(features)},
    )


def _check_no_target_leakage(frame: pd.DataFrame) -> CheckResult:
    """Guard against the target sneaking into the feature matrix."""
    features = [column for column in frame.columns if column != config.TARGET_COLUMN]
    if config.TARGET_COLUMN in features:
        return CheckResult(
            "no_target_leakage", False, SEVERITY_CRITICAL,
            "The target column is present in the feature list.",
        )
    suspicious = [
        column for column in features
        if column.strip().lower() in {"class", "label", "target", "y", "is_fraud", "fraud"}
    ]
    if suspicious:
        return CheckResult(
            "no_target_leakage", False, SEVERITY_CRITICAL,
            f"Feature column(s) {suspicious} look like target-derived columns.",
            {"suspicious": suspicious},
        )
    return CheckResult(
        "no_target_leakage", True, SEVERITY_CRITICAL,
        f"Target {config.TARGET_COLUMN!r} excluded from the {len(features)} features.",
    )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def validate_dataset(frame: pd.DataFrame | None) -> ValidationReport:
    """Run every check and return the aggregated report. Never raises."""
    report = ValidationReport()

    present = _check_present(frame)
    report.checks.append(present)
    if not present.passed:
        return report

    assert frame is not None  # narrowed by the check above
    report.checks.extend(
        [
            _check_not_empty(frame),
            _check_target_column(frame),
            _check_required_columns(frame),
            _check_missing_values(frame),
            _check_infinite_values(frame),
            _check_duplicates(frame),
            _check_numeric_features(frame),
            _check_no_target_leakage(frame),
        ]
    )
    # Target-value checks are only meaningful once the target exists.
    if any(check.passed for check in report.checks if check.name == "target_column_present"):
        report.checks.extend(
            [_check_target_values(frame), _check_target_distribution(frame)]
        )

    LOGGER.info(report.summary())
    return report


def enforce_valid_dataset(frame: pd.DataFrame | None) -> ValidationReport:
    """Validate and raise :class:`DataValidationError` if anything critical failed."""
    report = validate_dataset(frame)
    if not report.passed:
        failures = "\n  - ".join(
            f"{check.name}: {check.message}" for check in report.critical_failures
        )
        raise DataValidationError(
            f"Dataset validation failed ({len(report.critical_failures)} critical "
            f"problem(s)):\n  - {failures}"
        )
    return report


def save_report(report: ValidationReport, path: Path | None = None) -> Path:
    """Persist the report as JSON so it can be inspected after a DAG run."""
    destination = path or config.VALIDATION_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    LOGGER.info("Validation report written to %s", destination)
    return destination
