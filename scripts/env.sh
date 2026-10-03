#!/usr/bin/env bash
# Shared environment for every command in this project.
#
# Usage:   source scripts/env.sh
#
# The virtualenv is the one that already exists on this machine
# (~/.venvs/houseprice) and already contains the full pinned stack, so this
# project needs no additional installation. Override it by exporting
# VENV=/path/to/other/venv before sourcing this file.

# Resolve the project root from this script's location (no hardcoded paths).
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || return 1

# --- Virtualenv -----------------------------------------------------------
export VENV="${VENV:-$HOME/.venvs/houseprice}"
export PYTHON="$VENV/bin/python"
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# --- Ports ----------------------------------------------------------------
export MLFLOW_PORT="${MLFLOW_PORT:-5000}"
export AIRFLOW_PORT="${AIRFLOW_PORT:-8080}"

# --- MLflow ---------------------------------------------------------------
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:$MLFLOW_PORT}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-credit-card-fraud-detection}"

# --- Airflow (state lives inside the project, not in ~/airflow) -----------
export AIRFLOW_HOME="${AIRFLOW_HOME:-$PROJECT_ROOT/.airflow}"
export AIRFLOW__CORE__DAGS_FOLDER="$PROJECT_ROOT/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__EXECUTOR=LocalExecutor
export AIRFLOW__DAG_PROCESSOR__REFRESH_INTERVAL=5
# Keep Airflow's own chatter out of the pipeline logs.
export AIRFLOW__LOGGING__LOGGING_LEVEL=INFO

# --- Matplotlib must be headless inside Airflow workers -------------------
export MPLBACKEND=Agg

echo "PROJECT_ROOT      = $PROJECT_ROOT"
echo "VENV              = $VENV"
echo "python            = $($PYTHON --version 2>&1)"
echo "AIRFLOW_HOME      = $AIRFLOW_HOME"
echo "MLFLOW_TRACKING_URI = $MLFLOW_TRACKING_URI"
echo "MLFLOW experiment = $MLFLOW_EXPERIMENT_NAME"
echo "Airflow UI        = http://localhost:$AIRFLOW_PORT"
echo "MLflow UI         = http://localhost:$MLFLOW_PORT"
