from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_complete(output: Path) -> bool:
    final = read_json(output / "FINAL_DEVELOPMENT_STATUS.json", {})
    audit = read_json(output / "DEVELOPMENT_COMPLETION_AUDIT.json", {})
    lock = read_json(output / "LOCKED_3D5_RECIPE.json", {})
    return bool(
        final.get("status") == "completed"
        and audit.get("counts_ok") is True
        and int(audit.get("actual_model_fits", 0)) == int(audit.get("expected_model_fits", -1))
        and lock.get("status") == "LOCKED_FOR_FINAL_SEALED"
        and final.get("final_sealed_consumed") is False
    )


def export_result_zip(package_root: Path, output: Path) -> dict[str, Any]:
    desktop = Path.home() / "Desktop"
    desktop.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = desktop / f"CrashWatch_RegimeSeal_3D5_RESULTS_{stamp}.zip"
    final = read_json(output / "FINAL_DEVELOPMENT_STATUS.json", {})
    audit = read_json(output / "DEVELOPMENT_COMPLETION_AUDIT.json", {})
    recipe = read_json(output / "LOCKED_3D5_RECIPE.json", {})
    start_here = "\n".join(
        [
            "# CrashWatch RegimeSeal 3D5 FullLoad result package",
            "",
            f"- status: {final.get('status')}",
            f"- elapsed_hours: {float(final.get('elapsed_seconds', 0.0)) / 3600.0:.3f}",
            f"- model_fits: {audit.get('actual_model_fits')}/{audit.get('expected_model_fits')}",
            f"- locked_profile: {recipe.get('locked_profile')}",
            f"- LightGBM config: {recipe.get('lgb_config_id')}",
            f"- XGBoost config: {recipe.get('xgb_config_id')}",
            f"- XGBoost blend weight: {recipe.get('xgb_weight')}",
            "- sealed data was not used for this development selection.",
            "",
            "Read LOCKED_3D5_RECIPE.json, FINAL_DEVELOPMENT_STATUS.json,",
            "confirm_blend_grid.csv, SEARCH_TO_CONFIRM_SELECTION.json,",
            "and the search/confirm summaries first. Check regime-window and",
            "regime-metric CSVs before interpreting overall scores.",
            "",
        ]
    )
    files: list[tuple[Path, str]] = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.suffix.lower() in {".json", ".csv", ".txt", ".log"}:
            if path.name not in {"SUPERVISOR_FAILED.json", "STOP_SUPERVISOR"}:
                files.append((path, f"output/{path.name}"))
    for folder in ("search", "confirm"):
        source = output / folder
        if source.exists():
            for path in sorted(source.rglob("*")):
                if path.is_file() and path.suffix.lower() in {".json", ".csv", ".txt"}:
                    files.append((path, f"{folder}/{path.relative_to(source).as_posix()}"))
    for relative in [
        "config_regime_3d5_full_load.json",
        "README_KO.md",
        "EXPERIMENT_DESIGN_KO.md",
        "CODE_AUDIT.md",
        "run_regime_development.py",
        "supervise_regime_3d5.py",
    ]:
        path = package_root / relative
        if path.exists():
            files.append((path, f"code_context/{relative}"))
    for path in sorted((package_root / "cwregime").glob("*.py")):
        files.append((path, f"code_context/cwregime/{path.name}"))

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipped:
        zipped.writestr("00_START_HERE.md", start_here.encode("utf-8"))
        seen: set[str] = set()
        for source, arcname in files:
            if arcname not in seen:
                zipped.write(source, arcname)
                seen.add(arcname)
    with zipfile.ZipFile(archive, "r") as zipped:
        bad = zipped.testzip()
        names = set(zipped.namelist())
    required = {
        "00_START_HERE.md",
        "output/FINAL_DEVELOPMENT_STATUS.json",
        "output/DEVELOPMENT_COMPLETION_AUDIT.json",
        "output/LOCKED_3D5_RECIPE.json",
        "output/confirm_blend_grid.csv",
    }
    if bad is not None or not required.issubset(names):
        raise RuntimeError(f"result ZIP verification failed: bad={bad}, missing={sorted(required - names)}")
    digest = sha256(archive)
    checksum_path = archive.with_suffix(archive.suffix + ".sha256.txt")
    checksum_path.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    export = {
        "status": "completed",
        "archive": str(archive),
        "size_bytes": archive.stat().st_size,
        "size_mb": archive.stat().st_size / 1024**2,
        "sha256": digest,
        "entries": len(names),
        "verified": True,
        "created_epoch": time.time(),
    }
    (output / "RESULT_EXPORT_STATUS.json").write_text(json.dumps(export, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return export


def main() -> int:
    parser = argparse.ArgumentParser(description="Supervise RegimeSeal 3D5 development run and export a result ZIP")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-restarts", type=int, default=1)
    parser.add_argument("--retry-delay-seconds", type=float, default=20.0)
    args = parser.parse_args()
    package_root = Path(__file__).resolve().parent
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    stop_marker = output / "STOP_SUPERVISOR"
    command = [
        sys.executable,
        str(package_root / "run_regime_development.py"),
        "--project-root", str(Path(args.project_root).expanduser().resolve()),
        "--dataset", str(Path(args.dataset).expanduser().resolve()),
        "--output", str(output),
        "--config", str(Path(args.config).expanduser().resolve()),
    ]
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max(1, args.max_restarts + 1) + 1):
        if stop_marker.exists():
            return 3
        started = time.time()
        print(f"SUPERVISOR attempt={attempt} command={command}", flush=True)
        completed = subprocess.run(command, cwd=package_root, env=os.environ.copy(), check=False)
        record = {
            "attempt": attempt,
            "returncode": completed.returncode,
            "elapsed_seconds": time.time() - started,
            "result_complete": is_complete(output),
        }
        attempts.append(record)
        (output / "supervisor_attempts.json").write_text(json.dumps(attempts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if completed.returncode == 0 and record["result_complete"]:
            export = export_result_zip(package_root, output)
            print(json.dumps({"status": "completed", "export": export}, ensure_ascii=False), flush=True)
            return 0
        if attempt > args.max_restarts or stop_marker.exists():
            break
        print(f"SUPERVISOR retrying from atomic checkpoints in {args.retry_delay_seconds:.1f}s", flush=True)
        time.sleep(max(1.0, args.retry_delay_seconds))
    failure = {"status": "failed_after_retries", "attempts": attempts, "created_epoch": time.time()}
    (output / "SUPERVISOR_FAILED.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
