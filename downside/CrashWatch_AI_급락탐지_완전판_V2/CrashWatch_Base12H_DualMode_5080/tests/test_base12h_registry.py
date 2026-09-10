from pathlib import Path

from dual_ablation.base12h.runner import _select_affinity
from dual_ablation.base12h.registry import TaskRegistry


def test_mode_switch_preserves_block_backend_and_threads(tmp_path: Path):
    registry = TaskRegistry(tmp_path / "tasks.sqlite3")
    registry.ensure_blocks([("sig_fold00_seed17", 0, 17, 100)])
    first = registry.claim_next_block(
        worker_name="pubg", default_backend="cpu", default_threads=4, profile="pubg"
    )
    assert first is not None
    assert first.backend == "cpu"
    assert first.threads == 4
    registry.release_block(first.block_id, "switch mode")

    resumed = registry.claim_next_block(
        worker_name="full", default_backend="cuda", default_threads=8, profile="full"
    )
    assert resumed is not None
    assert resumed.backend == "cpu"
    assert resumed.threads == 4


def test_completed_task_survives_resume(tmp_path: Path):
    registry = TaskRegistry(tmp_path / "tasks.sqlite3")
    registry.ensure_blocks([("b", 0, 17, 1)])
    block = registry.claim_next_block(worker_name="w", default_backend="cpu", default_threads=4, profile="pubg")
    assert block is not None
    registry.start_task(task_id="t", block_id="b", experiment="baseline", backend="cpu", threads=4)
    registry.complete_task(task_id="t", result_path="m.parquet", record_path="r.json")
    registry.release_block("b")
    assert registry.task_completed("t")


def test_dataset_signature_change_can_register_same_fold_seed(tmp_path: Path):
    registry = TaskRegistry(tmp_path / "tasks.sqlite3")
    registry.ensure_blocks([("sig_a_fold00_seed17", 0, 17, 2)])
    registry.ensure_blocks([("sig_b_fold00_seed17", 0, 17, 1)])
    assert registry.status()["blocks_total"] == 2


def test_two_claims_get_different_blocks(tmp_path: Path):
    registry = TaskRegistry(tmp_path / "tasks.sqlite3")
    registry.ensure_blocks([("a", 0, 17, 2), ("b", 1, 17, 1)])
    first = registry.claim_next_block(worker_name="w1", default_backend="cuda", default_threads=8, profile="full")
    second = registry.claim_next_block(worker_name="w2", default_backend="cuda", default_threads=8, profile="full")
    assert first is not None and second is not None
    assert first.block_id != second.block_id


def test_affinity_selection_uses_original_eligible_set():
    eligible = list(range(16))
    assert _select_affinity(eligible, "pubg", 4, 0, 1) == [0, 2, 4, 6]
    assert _select_affinity(eligible, "full", 8, 0, 2) == list(range(8))
    assert _select_affinity(eligible, "full", 8, 1, 2) == list(range(8, 16))
    # Two full-mode workers resuming CPU4 blocks should not overlap.
    assert _select_affinity(eligible, "pubg", 4, 0, 2) == [0, 2, 4, 6]
    assert _select_affinity(eligible, "pubg", 4, 1, 2) == [8, 10, 12, 14]
