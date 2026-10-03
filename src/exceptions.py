"""Exceptions raised by the pipeline.

Every failure mode gets its own type so callers (and Airflow tasks) can react
to a specific cause instead of parsing error strings.
"""

from __future__ import annotations


class FraudPipelineError(Exception):
    """Base class for every error raised by this project."""


class DatasetError(FraudPipelineError):
    """The dataset could not be downloaded, cached or parsed."""


class DataValidationError(FraudPipelineError):
    """The dataset failed a validation check and must not be trained on."""


class PreprocessingError(FraudPipelineError):
    """Features/target could not be prepared or transformed."""


class TrainingError(FraudPipelineError):
    """A model could not be fitted."""


class EvaluationError(FraudPipelineError):
    """Metrics or evaluation artifacts could not be produced."""


class MLflowError(FraudPipelineError):
    """MLflow is unreachable, or logging to MLflow failed."""
