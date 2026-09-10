from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from ..config import get_paths
from ..io_utils import atomic_json
from .common import load_local_env, now_iso, sanitize_secret_error
from .dart_pit import collect_dart_point_in_time
from .krx_actual import collect_krx_actual_data
from .macro import collect_macro_credit
from .naver_flow import collect_naver_flow_fallback
from .stock_lending import collect_stock_lending
from .validation import validate_required_data

LOGGER = logging.getLogger(__name__)


def run_required_data_download(
    project: Path,
    start: str,
    end: str,
    *,
    sources: list[str],
    overwrite: bool = False,
    strict: bool = False,
    dart_documents: bool = True,
    naver_start: str | None = None,
    krx_chunk_months: int = 24,
    lending_chunk_months: int = 12,
) -> dict[str, Any]:
    """Download every requested source, then apply the strict gate once.

    A failed mandatory KRX session must not prevent the independent lending,
    DART, Naver fallback, and macro sources from being collected. All failures
    are persisted before strict validation raises.
    """
    load_local_env(project)
    paths = get_paths(project)
    root = paths.raw_dual / "required_data_v3"
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result: dict[str, Any] = {
        "created_at": now_iso(),
        "requested_start": start,
        "requested_end": end,
        "sources": sources,
        "strict_requested": strict,
        "steps": {},
    }

    step_functions = {
        "macro": lambda: collect_macro_credit(paths, start, end, overwrite=overwrite),
        "krx": lambda: collect_krx_actual_data(
            paths,
            start,
            end,
            overwrite=overwrite,
            chunk_months=krx_chunk_months,
        ),
        "lending": lambda: collect_stock_lending(
            paths, start, end, overwrite=overwrite, chunk_months=lending_chunk_months
        ),
        "dart": lambda: collect_dart_point_in_time(
            paths,
            start,
            end,
            overwrite=overwrite,
            download_documents=dart_documents,
        ),
        "naver": lambda: collect_naver_flow_fallback(
            paths,
            naver_start or max(start, "2020-01-01"),
            end,
            overwrite=overwrite,
        ),
    }

    for source in sources:
        if source not in step_functions:
            result["steps"][source] = {"status": "unknown_source"}
            continue
        LOGGER.info("필수 데이터 수집 단계 시작: %s", source)
        try:
            payload = step_functions[source]()
            result["steps"][source] = {"status": "completed", "summary": payload}
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("수집 단계 실패: %s", source)
            result["steps"][source] = {
                "status": "failed",
                "error": sanitize_secret_error(f"{type(exc).__name__}: {exc}"),
            }

    # Always persist a non-strict report first, even when the final gate fails.
    result["validation"] = validate_required_data(paths, strict=False)
    result["elapsed_seconds"] = time.perf_counter() - started
    result["completed_at"] = now_iso()
    atomic_json(result, root / "required_data_run_summary.json")

    if strict:
        # This raises only after all independent sources have had a chance to run.
        result["validation"] = validate_required_data(paths, strict=True)
        atomic_json(result, root / "required_data_run_summary.json")
    return result
