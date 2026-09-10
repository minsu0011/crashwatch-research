from __future__ import annotations

from dual_ablation.refine12h.registry import RefineTaskRegistry


def test_pubg_task_backend_is_preserved_for_full_resume(tmp_path) -> None:
    registry = RefineTaskRegistry(tmp_path / "tasks.sqlite")
    registry.ensure_tasks([{
        "task_id": "tree1", "stage": "tree", "family": "xgboost", "outer_fold": 0,
        "seed": 17, "priority": 100, "required_profile": "any", "payload": {},
    }])
    claim = registry.claim_next(profile="pubg", stages=("tree",), worker_name="pubg", backend="cpu", threads=4)
    assert claim is not None
    assert claim.backend == "cpu" and claim.threads == 4
    registry.release(claim.task_id, "switch mode")
    resumed = registry.claim_next(profile="full", stages=("tree",), worker_name="full", backend="cuda", threads=8)
    assert resumed is not None
    assert resumed.backend == "cpu"
    assert resumed.threads == 4


def test_deep_tasks_are_full_only(tmp_path) -> None:
    registry = RefineTaskRegistry(tmp_path / "tasks.sqlite")
    registry.ensure_tasks([{
        "task_id": "deep1", "stage": "deep", "family": "tcn", "outer_fold": 1,
        "seed": 17, "priority": 10, "required_profile": "full", "payload": {},
    }])
    assert registry.claim_next(profile="pubg", stages=("deep",), worker_name="pubg", backend="cpu", threads=4) is None
    assert registry.claim_next(profile="full", stages=("deep",), worker_name="full", backend="cuda", threads=12) is not None


def test_ensure_tasks_reprioritizes_only_pending_rows(tmp_path) -> None:
    registry = RefineTaskRegistry(tmp_path / "tasks.sqlite")
    task = {
        "task_id": "tree1", "stage": "tree", "family": "lightgbm", "outer_fold": 2,
        "seed": 17, "priority": 10, "required_profile": "any", "payload": {},
    }
    registry.ensure_tasks([task])
    registry.ensure_tasks([{**task, "priority": 99}])
    with registry.connect() as connection:
        assert connection.execute("SELECT priority FROM tasks WHERE task_id='tree1'").fetchone()["priority"] == 99

    claim = registry.claim_next(profile="full", stages=("tree",), worker_name="full", backend="cuda", threads=8)
    assert claim is not None
    registry.complete(claim.task_id)
    registry.ensure_tasks([{**task, "priority": 1}])
    with registry.connect() as connection:
        row = connection.execute("SELECT priority,status FROM tasks WHERE task_id='tree1'").fetchone()
    assert row["priority"] == 99
    assert row["status"] == "completed"
