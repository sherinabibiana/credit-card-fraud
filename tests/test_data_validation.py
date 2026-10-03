"""Tests for the validation gate.

The validation step is the project's safety net, so each failure mode named in
the specification gets its own test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_validation import (
    SEVERITY_CRITICAL,
    ValidationReport,
    enforce_valid_dataset,
    validate_dataset,
)
from src.exceptions import DataValidationError
from tests.conftest import make_frame


def _check(report: ValidationReport, name: str):
    """Fetch a single check result by name, failing the test if absent."""
    for check in report.checks:
        if check.name == name:
            return check
    raise AssertionError(f"check {name!r} was not run; got {[c.name for c in report.checks]}")


def _critical_failures(report: ValidationReport) -> set[str]:
    return {check.name for check in report.critical_failures}


# --------------------------------------------------------------------------
# happy path
# --------------------------------------------------------------------------
def test_valid_dataset_passes(valid_frame):
    report = validate_dataset(valid_frame)

    assert report.passed, report.summary()
    assert report.critical_failures == []
    assert len(report.checks) >= 10, "every documented check should run"


def test_valid_dataset_passes_enforce(valid_frame):
    report = enforce_valid_dataset(valid_frame)
    assert report.passed


def test_target_distribution_is_reported(valid_frame):
    report = validate_dataset(valid_frame)
    check = _check(report, "target_distribution")

    assert check.passed
    details = check.details
    assert details["n_legitimate"] + details["n_fraud"] == len(valid_frame)
    assert 0 < details["fraud_percentage"] < 100


def test_duplicates_are_reported_as_a_warning(frame_with_duplicates):
    report = validate_dataset(frame_with_duplicates)
    check = _check(report, "duplicate_rows")

    assert check.passed, "duplicates are allowed, but must be surfaced"
    assert check.severity != SEVERITY_CRITICAL
    assert check.details["n_duplicate_rows_exact"] > 0


# --------------------------------------------------------------------------
# failure modes
# --------------------------------------------------------------------------
def test_missing_dataset_fails():
    report = validate_dataset(None)

    assert not report.passed
    assert "dataset_present" in _critical_failures(report)


def test_empty_dataset_fails():
    report = validate_dataset(pd.DataFrame(columns=config.EXPECTED_COLUMNS))

    assert not report.passed
    assert "dataset_not_empty" in _critical_failures(report)


def test_dataset_below_minimum_rows_fails():
    frame = make_frame(n_rows=50)
    report = validate_dataset(frame)

    assert not report.passed
    assert "dataset_not_empty" in _critical_failures(report)


def test_missing_target_column_fails(valid_frame):
    frame = valid_frame.drop(columns=[config.TARGET_COLUMN])
    report = validate_dataset(frame)

    assert not report.passed
    assert "target_column_present" in _critical_failures(report)


def test_missing_required_feature_column_fails(valid_frame):
    frame = valid_frame.drop(columns=["V17"])
    report = validate_dataset(frame)

    assert not report.passed
    assert "required_columns_present" in _critical_failures(report)
    assert "V17" in _check(report, "required_columns_present").message


def test_invalid_target_values_fail(valid_frame):
    frame = valid_frame.copy()
    frame.loc[0, config.TARGET_COLUMN] = 7
    report = validate_dataset(frame)

    assert not report.passed
    assert "target_values_valid" in _critical_failures(report)


def test_target_with_only_one_class_fails():
    frame = make_frame()
    frame[config.TARGET_COLUMN] = 0
    report = validate_dataset(frame)

    assert not report.passed
    assert "target_distribution" in _critical_failures(report)


def test_non_numeric_feature_fails(valid_frame):
    frame = valid_frame.copy()
    frame["V3"] = "not-a-number"
    report = validate_dataset(frame)

    assert not report.passed
    assert "features_numeric" in _critical_failures(report)


def test_missing_values_within_tolerance_pass(valid_frame):
    """A handful of holes must not stop the pipeline - the imputer handles them."""
    frame = valid_frame.copy()
    frame.loc[0:4, "V5"] = np.nan

    report = validate_dataset(frame)
    check = _check(report, "missing_values")
    assert check.passed
    assert check.details["missing_total"] == 5


def test_excessive_missing_values_fail(valid_frame, monkeypatch):
    """Exceeding config.MAX_MISSING_RATIO is a critical failure."""
    monkeypatch.setattr(config, "MAX_MISSING_RATIO", 0.01)
    frame = valid_frame.copy()
    frame.loc[:, "V5"] = np.nan

    report = validate_dataset(frame)
    assert not report.passed
    assert "missing_values" in _critical_failures(report)
    assert check_ratio_is_reported(report)


def check_ratio_is_reported(report: ValidationReport) -> bool:
    """The failure message must state the measured missing ratio."""
    check = _check(report, "missing_values")
    return "missing values" in check.message and "exceed" in check.message


def test_target_derived_feature_column_is_rejected(valid_frame):
    frame = valid_frame.copy()
    frame["is_fraud"] = frame[config.TARGET_COLUMN]
    report = validate_dataset(frame)

    check = _check(report, "no_target_leakage")
    assert not check.passed
    assert check.is_blocking


def test_enforce_raises_with_all_failures_listed():
    frame = make_frame(n_rows=10).drop(columns=[config.TARGET_COLUMN])

    with pytest.raises(DataValidationError) as excinfo:
        enforce_valid_dataset(frame)

    message = str(excinfo.value)
    assert "validation failed" in message
    assert "target_column_present" in message


def test_report_serialises_to_json(valid_frame):
    import json

    payload = json.dumps(validate_dataset(valid_frame).to_dict(), default=str)
    assert "checks" in payload
