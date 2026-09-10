from pathlib import Path

from dual_ablation.refine12h.reporting import _read_csv_or_empty


def test_headerless_empty_csv_is_treated_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("\r\n", encoding="utf-8-sig")

    assert _read_csv_or_empty(path).empty
