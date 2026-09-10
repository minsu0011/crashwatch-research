from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import priority_data_downloader as mod


def test_parse_years():
    assert mod.parse_years("2018-2020,2025") == {2018, 2019, 2020, 2025}


def test_chunk_years():
    chunks = list(mod.chunk_years("2019-06-01", "2020-02-01"))
    assert chunks == [("20190601", "20191231"), ("20200101", "20200201")]


def test_extract_items():
    payload = {"response": {"body": {"totalCount": 1, "items": {"item": {"a": 1}}}}}
    items, total = mod.extract_items(payload)
    assert total == 1
    assert items == [{"a": 1}]


def test_read_tickers(tmp_path: Path):
    path = tmp_path / "tickers.csv"
    pd.DataFrame({"ticker": [5930, "000660"], "enabled": [1, 1]}).to_csv(path, index=False)
    assert mod.read_tickers(path) == ["000660", "005930"]
