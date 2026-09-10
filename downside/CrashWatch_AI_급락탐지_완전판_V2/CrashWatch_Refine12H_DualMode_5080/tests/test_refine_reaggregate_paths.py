from pathlib import Path

from dual_ablation.refine12h.reaggregate import _resolve_existing


def test_mojibake_absolute_path_recovers_unambiguous_source_subtree(tmp_path: Path) -> None:
    source = tmp_path / "ablation_base12h"
    prediction = source / "prediction_cache" / "outer" / "cpu" / "same_hash.parquet"
    metrics = source / "task_metrics" / "same_hash.parquet"
    prediction.parent.mkdir(parents=True)
    metrics.parent.mkdir(parents=True)
    prediction.write_bytes(b"prediction")
    metrics.write_bytes(b"metrics")

    broken_prefix = "C:/Users/example/CrashWatch_AI_湲됰씫?먯?"
    prediction_record = broken_prefix + "/ablation_base12h/prediction_cache/outer/cpu/same_hash.parquet"
    metrics_record = broken_prefix + "/ablation_base12h/task_metrics/same_hash.parquet"

    assert _resolve_existing(prediction_record, source, "same_hash.parquet") == prediction
    assert _resolve_existing(metrics_record, source, "same_hash.parquet") == metrics
