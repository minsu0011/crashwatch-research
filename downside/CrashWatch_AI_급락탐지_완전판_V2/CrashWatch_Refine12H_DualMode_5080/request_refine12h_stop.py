from __future__ import annotations

import argparse
from pathlib import Path

from dual_ablation.config import get_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--result-dir", type=Path, default=None)
    args = parser.parse_args()
    paths = get_paths(args.project)
    result_dir = (args.result_dir or paths.data_root / "refine12h").resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    flag = result_dir / "REQUEST_SAFE_STOP.flag"
    flag.write_text("user requested safe stop", encoding="utf-8")
    print(f"safe stop requested: {flag}")


if __name__ == "__main__":
    main()
