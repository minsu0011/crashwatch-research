#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_ablation.config import get_paths
from dual_ablation.data_acquisition.krx_csv_import import import_licensed_krx_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="사용자가 합법적으로 내려받은 KRX CSV를 Finance11H에 통합")
    parser.add_argument("input_dir")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ratio-unit", choices=["percent", "fraction", "auto"], default="percent", help="공식 KRX CSV는 percent 권장")
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    result = import_licensed_krx_csv(get_paths(project), Path(args.input_dir).resolve(), overwrite=args.overwrite, ratio_unit=args.ratio_unit)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
