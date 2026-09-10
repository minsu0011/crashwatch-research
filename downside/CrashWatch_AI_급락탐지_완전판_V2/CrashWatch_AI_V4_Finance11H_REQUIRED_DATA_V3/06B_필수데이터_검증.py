#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_ablation.config import get_paths
from dual_ablation.data_acquisition.validation import validate_required_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    paths = get_paths(Path(__file__).resolve().parent)
    print(json.dumps(validate_required_data(paths, strict=args.strict), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
