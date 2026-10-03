"""Tests for the Airflow DAG definition itself.

These import the DAG file directly, so they verify the task graph without
needing a running Airflow instance.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

DAG_FILE = Path(__file__).resolve().parents[1] / "dags" / "fraud_detection_dag.py"

EXPECTED_CHAIN = [
    "load_data",
    "validate_data",
    "prepare_data",
    "split_data",
    "train_models",
    "evaluate_models",
    "log_results",
]


@pytest.fixture(scope="module")
def dag_module():
    if not DAG_FILE.exists():
        pytest.skip(f"DAG file not found at {DAG_FILE}")
    spec = importlib.util.spec_from_file_location("fraud_detection_dag_under_test", DAG_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dag(dag_module):
    return dag_module.dag


def test_dag_id(dag):
    assert dag.dag_id == "credit_card_fraud_detection"


def test_dag_has_no_automatic_schedule(dag):
    """The pipeline is triggered by hand so runs are deliberate."""
    assert dag.schedule is None
    assert dag.catchup is False


def test_all_expected_tasks_exist(dag):
    for task_id in EXPECTED_CHAIN:
        assert task_id in dag.task_dict, f"missing task {task_id}"


def test_task_graph_is_a_linear_chain(dag):
    for current, nxt in zip(EXPECTED_CHAIN, EXPECTED_CHAIN[1:]):
        downstream = {t.task_id for t in dag.task_dict[current].downstream_list}
        assert nxt in downstream, f"{current} must come before {nxt}"

    assert dag.task_dict["load_data"].upstream_list == []
    assert dag.task_dict["log_results"].downstream_list == []


def test_evaluate_models_depends_on_both_train_and_split(dag):
    upstream = {t.task_id for t in dag.task_dict["evaluate_models"].upstream_list}
    assert upstream == {"train_models", "split_data"}


def test_dag_is_documented_and_tagged(dag):
    assert dag.description
    assert dag.doc_md, "doc_md renders the module docstring in the Airflow UI"
    # dag.tags is a set, so membership must be checked per tag, not by substring
    assert any("fraud" in tag for tag in dag.tags)
    assert any("mlflow" in tag for tag in dag.tags)


def test_every_task_is_retried_once(dag):
    for task in dag.task_dict.values():
        assert task.retries == 1, f"{task.task_id} should retry transient failures"


def test_task_bodies_never_call_a_name_that_is_also_a_task_id(dag_module):
    """A task body must not reference any name shadowed by a @task definition.

    ``@task`` rebinds the decorated function's name in the DAG module. So if the
    body of task ``split_data`` calls ``split_data(...)`` intending the imported
    ``src.preprocessing.split_data``, it actually calls the task object and
    fails at runtime with "too many positional arguments". Import such helpers
    under an alias so no task body refers to a task id.
    """
    task_ids = set(dag_module.dag.task_dict)

    offenders: list[tuple[str, list[str]]] = []
    for task_id, task in dag_module.dag.task_dict.items():
        func = getattr(task, "python_callable", None)
        code = getattr(func, "__code__", None)
        if code is None:
            continue
        shadowed = sorted(set(code.co_names) & task_ids)
        if shadowed:
            offenders.append((task_id, shadowed))

    assert not offenders, (
        f"task bodies reference shadowed task name(s): {offenders}; "
        "import the helper under an alias instead"
    )


def test_split_task_uses_the_real_splitting_function(dag_module):
    """Pin the exact fix: the task body must reach the real function."""
    from src.preprocessing import split_data as real_split_data

    assert dag_module.split_dataset is real_split_data
    assert dag_module.split_data is not real_split_data
