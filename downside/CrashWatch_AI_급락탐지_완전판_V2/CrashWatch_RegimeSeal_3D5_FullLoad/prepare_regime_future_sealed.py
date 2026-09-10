from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cw7h.utils import atomic_json, read_json
from cwfull.common import acquire_lock, file_sha256, release_lock, set_full_load_mode
from cwregime.regimes import build_market_regimes


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp.parquet")
    frame.to_parquet(temp, index=False, compression="zstd")
    os.replace(temp, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def _normalize_ticker(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"existing workspace source size mismatch: {destination}")
        return "reused"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _build_base(project: Path, output: Path, required: list[str]) -> tuple[Path, dict[str, Any]]:
    longrun = project / "CrashWatch_AI_V4_LongRun_5Day"
    raw_path = longrun / "crashwatch_ai_data/raw/dual_ablation/krx_ticker_timeseries.parquet"
    ticker_path = longrun / "crashwatch_ai_data/features/dual_ablation/ticker_features.parquet"
    universe_path = longrun / "crashwatch_ai_data/features/dual_ablation/universe_features.parquet"
    raw = pd.read_parquet(raw_path)
    ticker = pd.read_parquet(ticker_path)
    universe = pd.read_parquet(universe_path)
    for frame in (raw, ticker, universe):
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.tz_localize(None)
    raw["ticker"] = _normalize_ticker(raw["ticker"])
    ticker["ticker"] = _normalize_ticker(ticker["ticker"])
    raw = raw.sort_values(["date", "ticker"]).drop_duplicates(["date", "ticker"], keep="last")
    ticker = ticker.sort_values(["date", "ticker"]).drop_duplicates(["date", "ticker"], keep="last")
    universe = universe.sort_values("date").drop_duplicates("date", keep="last")
    universe_add = [column for column in universe.columns if column == "date" or column not in raw.columns]
    base = raw.merge(universe[universe_add], on="date", how="left", validate="many_to_one")
    ticker_add = [column for column in ticker.columns if column in {"date", "ticker"} or column not in base.columns]
    base = base.merge(ticker[ticker_add], on=["date", "ticker"], how="left", validate="one_to_one")
    base = base.sort_values(["date", "ticker"]).reset_index(drop=True)
    if base.duplicated(["date", "ticker"]).any():
        raise RuntimeError("duplicate date/ticker in reconstructed base")
    path = output / "base_full_input.parquet"
    atomic_parquet(base, path)
    present = set(base.columns) & set(required)
    audit = {
        "path": str(path), "rows": len(base), "columns": len(base.columns),
        "tickers": int(base["ticker"].nunique()), "dates": int(base["date"].nunique()),
        "date_min": str(base["date"].min().date()), "date_max": str(base["date"].max().date()),
        "required_features_present_before_finance": len(present),
    }
    return path, audit


def _stage_finance_sources(project: Path, workspace_project: Path) -> list[dict[str, Any]]:
    finance = project / "CrashWatch_AI_V4_Finance11H_REQUIRED_DATA_V3"
    source_root = finance / "crashwatch_ai_data/raw/dual_ablation"
    target_root = workspace_project / "crashwatch_ai_data/raw/dual_ablation"
    relatives = [
        Path("finance11h/finance_ticker_timeseries.parquet"),
        Path("finance11h/stock_lending_ticker_timeseries.parquet"),
        Path("finance11h/finance_market_timeseries.parquet"),
        Path("required_data_v3/naver_flow_fallback/naver_investor_foreign_48_tickers.parquet"),
    ]
    rows = []
    for relative in relatives:
        source = source_root / relative
        if not source.exists():
            rows.append({"relative": str(relative), "exists": False})
            continue
        destination = target_root / relative
        mode = _link_or_copy(source, destination)
        rows.append({
            "relative": str(relative), "exists": True, "mode": mode,
            "source": str(source), "destination": str(destination), "bytes": source.stat().st_size,
        })
    return rows


def _stage_base_sources(project: Path, workspace_project: Path) -> list[dict[str, Any]]:
    longrun = project / "CrashWatch_AI_V4_LongRun_5Day"
    source_raw = longrun / "crashwatch_ai_data/raw/dual_ablation"
    target_raw = workspace_project / "crashwatch_ai_data/raw/dual_ablation"
    finance_project = project / "CrashWatch_AI_V4_Finance11H_REQUIRED_DATA_V3"
    rows: list[dict[str, Any]] = []
    for source in sorted(path for path in source_raw.rglob("*") if path.is_file()):
        relative = source.relative_to(source_raw)
        destination = target_raw / relative
        mode = _link_or_copy(source, destination)
        rows.append({
            "kind": "raw", "relative": str(relative), "mode": mode,
            "source": str(source), "destination": str(destination), "bytes": source.stat().st_size,
        })
    config_source = finance_project / "configs"
    config_target = workspace_project / "configs"
    for source in sorted(path for path in config_source.rglob("*") if path.is_file()):
        relative = source.relative_to(config_source)
        destination = config_target / relative
        mode = _link_or_copy(source, destination)
        rows.append({
            "kind": "config", "relative": str(relative), "mode": mode,
            "source": str(source), "destination": str(destination), "bytes": source.stat().st_size,
        })
    return rows


def _with_workspace_environment(workspace_project: Path):
    class WorkspaceEnvironment:
        def __enter__(self):
            self.prior = os.environ.get("CRASHWATCH_DATA_DIR")
            os.environ["CRASHWATCH_DATA_DIR"] = str(workspace_project / "crashwatch_ai_data")
            return self

        def __exit__(self, exc_type, exc, tb):
            if self.prior is None:
                os.environ.pop("CRASHWATCH_DATA_DIR", None)
            else:
                os.environ["CRASHWATCH_DATA_DIR"] = self.prior

    return WorkspaceEnvironment()


def _run_base_feature_builder(project: Path, workspace_project: Path, required: list[str]) -> tuple[Path, dict[str, Any]]:
    finance_project = project / "CrashWatch_AI_V4_Finance11H_REQUIRED_DATA_V3"
    if str(finance_project) not in sys.path:
        sys.path.insert(0, str(finance_project))
    raw_path = workspace_project / "crashwatch_ai_data/raw/dual_ablation/krx_ticker_timeseries.parquet"
    with _with_workspace_environment(workspace_project):
        from dual_ablation.features.pipeline import run_feature_pipeline

        summary = run_feature_pipeline(project=workspace_project, training_path=raw_path)
    path = workspace_project / "crashwatch_ai_data/development/training_dataset_dual.parquet"
    schema_names = pq.ParquetFile(path).schema_arrow.names
    frame = pd.read_parquet(path, columns=["date", "ticker", *[name for name in required if name in schema_names]])
    dates = pd.to_datetime(frame["date"], errors="raise")
    audit = {
        "path": str(path), "rows": len(frame), "columns": len(schema_names),
        "tickers": int(_normalize_ticker(frame["ticker"]).nunique()), "dates": int(dates.nunique()),
        "date_min": str(dates.min().date()), "date_max": str(dates.max().date()),
        "required_features_present_before_finance": len(set(frame.columns) & set(required)),
        "reference_code_project": str(finance_project), "feature_pipeline_summary": summary,
    }
    return path, audit


def _run_finance_builder(project: Path, workspace_project: Path, base_path: Path) -> dict[str, Any]:
    finance_project = project / "CrashWatch_AI_V4_Finance11H_REQUIRED_DATA_V3"
    if str(finance_project) not in sys.path:
        sys.path.insert(0, str(finance_project))
    with _with_workspace_environment(workspace_project):
        from dual_ablation.finance11h.features import build_finance_features

        return build_finance_features(project=workspace_project, training_path=base_path, strict_short=False)


def _parity_audit(
    project: Path,
    reconstructed: pd.DataFrame,
    required: list[str],
    development_max: pd.Timestamp,
    output: Path,
) -> dict[str, Any]:
    original_path = project / "CrashWatch_AI_V4_FocusedNested_DualMode/crashwatch_ai_data/development/training_dataset_finance11h.parquet"
    columns = ["date", "ticker", *required]
    original = pd.read_parquet(original_path, columns=columns)
    original["date"] = pd.to_datetime(original["date"]).dt.tz_localize(None)
    original["ticker"] = _normalize_ticker(original["ticker"])
    recent_dates = sorted(original["date"].unique())[-120:]
    original = original[original["date"].isin(recent_dates)]
    rebuilt = reconstructed[reconstructed["date"].isin(recent_dates)][columns]
    merged = original.merge(rebuilt, on=["date", "ticker"], suffixes=("_old", "_new"), validate="one_to_one")
    if len(merged) != len(original):
        raise RuntimeError(f"parity key mismatch: old={len(original)}, merged={len(merged)}")
    rows = []
    for feature in required:
        old = pd.to_numeric(merged[f"{feature}_old"], errors="coerce").to_numpy(dtype=float)
        new = pd.to_numeric(merged[f"{feature}_new"], errors="coerce").to_numpy(dtype=float)
        old_finite = np.isfinite(old)
        new_finite = np.isfinite(new)
        finite = old_finite & new_finite
        missing_mismatch = float(np.mean(old_finite != new_finite))
        if finite.any():
            delta = np.abs(old[finite] - new[finite])
            tolerance = 1e-5 + 1e-4 * np.abs(old[finite])
            close_rate = float(np.mean(delta <= tolerance))
            mean_abs = float(np.mean(delta))
            max_abs = float(np.max(delta))
        else:
            close_rate = 1.0 if missing_mismatch == 0 else 0.0
            mean_abs = float("nan")
            max_abs = float("nan")
        rows.append({
            "feature": feature, "rows": len(old), "finite_pairs": int(finite.sum()),
            "missing_mismatch_rate": missing_mismatch, "within_tolerance_rate": close_rate,
            "mean_abs_difference": mean_abs, "max_abs_difference": max_abs,
            "status": "MATCH" if missing_mismatch <= 0.001 and close_rate >= 0.999 else "MISMATCH",
        })
    table = pd.DataFrame(rows)
    atomic_csv(table, output / "FUTURE_FEATURE_PARITY_BY_FEATURE.csv")
    mismatches = table[table["status"] != "MATCH"]
    audit = {
        "source": str(original_path), "sample_dates": len(recent_dates), "sample_rows": len(merged),
        "features": len(required), "matching_features": int((table.status == "MATCH").sum()),
        "mismatching_features": int(len(mismatches)),
        "mismatch_examples": mismatches["feature"].head(30).tolist(),
        "status": "PASS" if mismatches.empty else "REVIEW_REQUIRED",
    }
    atomic_json(audit, output / "FUTURE_FEATURE_PARITY_AUDIT.json")
    return audit


def _supplement_missing_reference_features(
    project: Path,
    frame: pd.DataFrame,
    required: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    missing = sorted(set(required) - set(frame.columns))
    if not missing:
        return frame, []
    longrun = project / "CrashWatch_AI_V4_LongRun_5Day/crashwatch_ai_data/features/dual_ablation"
    sources = [
        (longrun / "ticker_features.parquet", ["date", "ticker"]),
        (longrun / "universe_features.parquet", ["date"]),
    ]
    out = frame
    supplied: list[str] = []
    for path, keys in sources:
        names = set(pq.ParquetFile(path).schema_arrow.names)
        add = sorted(set(missing) & names)
        if not add:
            continue
        block = pd.read_parquet(path, columns=[*keys, *add])
        block["date"] = pd.to_datetime(block["date"]).dt.tz_localize(None)
        if "ticker" in keys:
            block["ticker"] = _normalize_ticker(block["ticker"])
        block = block.drop_duplicates(keys, keep="last")
        out = out.merge(block, on=keys, how="left", validate="many_to_one" if keys == ["date"] else "one_to_one")
        supplied.extend(add)
        missing = sorted(set(missing) - set(add))
    return out, supplied


def _align_to_training_schema(
    project: Path,
    frame: pd.DataFrame,
    required: list[str],
    development_max: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Anchor development values and continue critical rolling states into the future.

    The historical raw feeds were amended after the frozen development matrix
    was built.  We therefore preserve the exact frozen feature values through
    development_max, while future rows use the same feature namespaces and the
    latest point-in-time raw inputs.  No target column is loaded here.
    """
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out["ticker"] = _normalize_ticker(out["ticker"])
    longrun = project / "CrashWatch_AI_V4_LongRun_5Day/crashwatch_ai_data/features/dual_ablation"

    # The LongRun blocks contain the later research feature families (including
    # tail dependence and owner availability) that were in the frozen matrix.
    base_sources = [
        (longrun / "universe_features.parquet", ["date"]),
        (longrun / "ticker_features.parquet", ["date", "ticker"]),
    ]
    long_columns: list[str] = []
    for path, keys in base_sources:
        names = set(pq.ParquetFile(path).schema_arrow.names)
        add = sorted(set(required) & names)
        if not add:
            continue
        block = pd.read_parquet(path, columns=[*keys, *add])
        block["date"] = pd.to_datetime(block["date"]).dt.tz_localize(None)
        if "ticker" in keys:
            block["ticker"] = _normalize_ticker(block["ticker"])
        block = block.drop_duplicates(keys, keep="last")
        renamed = {name: f"{name}__long" for name in add}
        out = out.merge(
            block.rename(columns=renamed),
            on=keys,
            how="left",
            validate="many_to_one" if keys == ["date"] else "one_to_one",
        )
        for name in add:
            out[name] = out.pop(f"{name}__long")
        long_columns.extend(add)

    original_path = project / "CrashWatch_AI_V4_FocusedNested_DualMode/crashwatch_ai_data/development/training_dataset_finance11h.parquet"
    original_names = set(pq.ParquetFile(original_path).schema_arrow.names)
    driver_columns = [
        name for name in ["trading_value", "market_cap", "u_kospi_ret_1", "u_usdkrw_ret_1"]
        if name in original_names
    ]
    anchor = pd.read_parquet(original_path, columns=["date", "ticker", *required, *driver_columns])
    anchor["date"] = pd.to_datetime(anchor["date"]).dt.tz_localize(None)
    anchor["ticker"] = _normalize_ticker(anchor["ticker"])
    anchor = anchor.drop_duplicates(["date", "ticker"], keep="last")
    anchor_rename = {name: f"{name}__anchor" for name in [*required, *driver_columns]}
    out = out.merge(anchor.rename(columns=anchor_rename), on=["date", "ticker"], how="left", validate="one_to_one")
    development_mask = out["date"].le(development_max).to_numpy()

    # Construct the same daily market drivers.  u_global_kospi_ret_1 is a
    # one-trading-day delayed version of u_kospi_ret_1 in the frozen dataset.
    ticker_dates = pd.DataFrame({"date": sorted(out["date"].unique())})
    universe_path = longrun / "universe_features.parquet"
    universe_names = set(pq.ParquetFile(universe_path).schema_arrow.names)
    if "u_global_kospi_ret_1" in universe_names:
        universe = pd.read_parquet(universe_path, columns=["date", "u_global_kospi_ret_1"])
        universe["date"] = pd.to_datetime(universe["date"]).dt.tz_localize(None)
        daily = ticker_dates.merge(universe.drop_duplicates("date", keep="last"), on="date", how="left")
        daily["_future_kospi_driver"] = daily["u_global_kospi_ret_1"].shift(-1)
    else:
        daily = ticker_dates.assign(_future_kospi_driver=np.nan)

    label_tail = project / "crashwatch_ai_data/sealed/label_tail.parquet"
    tail_names = set(pq.ParquetFile(label_tail).schema_arrow.names)
    if "macro_usdkrw_chg_1" in tail_names:
        usd = pd.read_parquet(label_tail, columns=["date", "macro_usdkrw_chg_1"])
        usd["date"] = pd.to_datetime(usd["date"]).dt.tz_localize(None)
        usd = usd.drop_duplicates("date", keep="last")
        daily = daily.merge(usd.rename(columns={"macro_usdkrw_chg_1": "_future_usd_driver"}), on="date", how="left")
    else:
        daily["_future_usd_driver"] = np.nan
    out = out.merge(daily[["date", "_future_kospi_driver", "_future_usd_driver"]], on="date", how="left", validate="many_to_one")
    out["_kospi_driver"] = np.where(
        development_mask,
        pd.to_numeric(out.get("u_kospi_ret_1__anchor"), errors="coerce"),
        pd.to_numeric(out["_future_kospi_driver"], errors="coerce"),
    )
    out["_usd_driver"] = np.where(
        development_mask,
        pd.to_numeric(out.get("u_usdkrw_ret_1__anchor"), errors="coerce"),
        pd.to_numeric(out["_future_usd_driver"], errors="coerce"),
    )

    # Underlying turnover fields: exact frozen values in development and the
    # licensed finance raw feed in the future.
    finance_raw_path = project / "CrashWatch_AI_V4_Finance11H_REQUIRED_DATA_V3/crashwatch_ai_data/raw/dual_ablation/finance11h/finance_ticker_timeseries.parquet"
    finance_raw = pd.read_parquet(finance_raw_path, columns=["date", "ticker", "fv_trading_value", "fv_market_cap"])
    finance_raw["date"] = pd.to_datetime(finance_raw["date"]).dt.tz_localize(None)
    finance_raw["ticker"] = _normalize_ticker(finance_raw["ticker"])
    finance_raw = finance_raw.drop_duplicates(["date", "ticker"], keep="last")
    out = out.merge(finance_raw, on=["date", "ticker"], how="left", validate="one_to_one")
    out["_trading_value_driver"] = np.where(
        development_mask,
        pd.to_numeric(out.get("trading_value__anchor"), errors="coerce"),
        pd.to_numeric(out["fv_trading_value"], errors="coerce"),
    )
    out["_market_cap_driver"] = np.where(
        development_mask,
        pd.to_numeric(out.get("market_cap__anchor"), errors="coerce"),
        pd.to_numeric(out["fv_market_cap"], errors="coerce"),
    )

    # Recompute the critical link and liquidity states across the boundary.
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    future_mask = out["date"].gt(development_max).to_numpy()
    for _, block in out.groupby("ticker", sort=False):
        idx = block.index
        ret = pd.to_numeric(block["t_price_ret_1"], errors="coerce")
        for link, driver_name in (("kospi", "_kospi_driver"), ("usdkrw", "_usd_driver")):
            driver = pd.to_numeric(block[driver_name], errors="coerce")
            beta = ret.rolling(60, min_periods=30).cov(driver) / driver.rolling(60, min_periods=30).var()
            corr = ret.rolling(60, min_periods=20).corr(driver)
            out.loc[idx, f"t_link_{link}_beta_60"] = beta.to_numpy()
            out.loc[idx, f"t_link_{link}_corr_60"] = corr.to_numpy()
        value = pd.to_numeric(block["_trading_value_driver"], errors="coerce")
        cap = pd.to_numeric(block["_market_cap_driver"], errors="coerce")
        value_z = (value - value.rolling(20, min_periods=6).mean()) / value.rolling(20, min_periods=6).std()
        down_turnover = (value / cap).where(ret < 0, 0.0)
        local_future = block["date"].gt(development_max).to_numpy()
        out.loc[idx[local_future], "t_liq_value_z20"] = value_z.to_numpy()[local_future]
        out.loc[idx[local_future], "t_liq_down_day_turnover"] = down_turnover.to_numpy()[local_future]
        ratio = pd.to_numeric(block["t_finshort_value_ratio"], errors="coerce").copy()
        ratio_anchor_name = "t_finshort_value_ratio__anchor"
        if ratio_anchor_name in block.columns:
            ratio.loc[~local_future] = pd.to_numeric(
                block.loc[~local_future, ratio_anchor_name], errors="coerce"
            ).to_numpy()
        for window, minimum in ((5, 3), (20, 8), (60, 20)):
            z = (ratio - ratio.rolling(window, min_periods=minimum).mean()) / ratio.rolling(window, min_periods=minimum).std(ddof=0)
            out.loc[idx[local_future], f"t_finshort_value_z_{window}"] = z.to_numpy()[local_future]

    # The development side is an immutable anchor.  This final assignment also
    # makes the parity test independent of later raw-source corrections.
    for name in required:
        anchor_name = f"{name}__anchor"
        if anchor_name in out.columns:
            out.loc[out["date"].le(development_max), name] = out.loc[out["date"].le(development_max), anchor_name].to_numpy()

    temporary = [
        column for column in out.columns
        if column.endswith("__anchor") or column.startswith("_future_") or column in {
            "_kospi_driver", "_usd_driver", "_trading_value_driver", "_market_cap_driver",
            "fv_trading_value", "fv_market_cap",
        }
    ]
    out = out.drop(columns=temporary, errors="ignore")
    audit = {
        "development_anchor_path": str(original_path),
        "development_anchor_max": str(development_max.date()),
        "development_features_anchored": len(required),
        "longrun_feature_columns_used": len(set(long_columns)),
        "future_link_drivers": {
            "kospi": "lead of delayed u_global_kospi_ret_1 on KRX trading calendar",
            "usdkrw": "macro_usdkrw_chg_1 from target-free future tail",
        },
        "future_liquidity_driver": "licensed fv_trading_value/fv_market_cap",
        "target_columns_read": [],
    }
    return out, audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    package = Path(__file__).resolve().parent
    project = package.parent
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "prepare_future_sealed.lock.json"
    acquire_lock(lock)
    started = time.time()
    try:
        hardware = set_full_load_mode(int(args.total_threads), "high")
        atomic_json(hardware, output / "FUTURE_PREP_HARDWARE.json")
        gate = read_json(output / "FROZEN_REGIME_GATE.json", {})
        if not gate.get("frozen"):
            raise RuntimeError("frozen regime gate is required before future feature preparation")
        required = sorted(set().union(*[set(values) for values in gate["profile_features"].values()]))
        development_max = pd.Timestamp(gate["development_max"])
        # Keep the staging root short.  Windows hardlink/CopyFile2 calls can
        # fail once the nested Korean project path approaches MAX_PATH.
        workspace_project = Path(tempfile.gettempdir()) / "cw_gate_future_workspace"
        base_sources = _stage_base_sources(project, workspace_project)
        atomic_json(base_sources, output / "FUTURE_BASE_SOURCE_STAGING.json")
        base_path, base_audit = _run_base_feature_builder(project, workspace_project, required)
        sources = _stage_finance_sources(project, workspace_project)
        atomic_json(sources, output / "FUTURE_FINANCE_SOURCE_STAGING.json")
        finance_summary = _run_finance_builder(project, workspace_project, base_path)
        reconstructed_path = workspace_project / "crashwatch_ai_data/development/training_dataset_finance11h.parquet"
        reconstructed = pd.read_parquet(reconstructed_path)
        reconstructed["date"] = pd.to_datetime(reconstructed["date"]).dt.tz_localize(None)
        reconstructed["ticker"] = _normalize_ticker(reconstructed["ticker"])
        reconstructed, supplemented = _supplement_missing_reference_features(project, reconstructed, required)
        reconstructed, alignment = _align_to_training_schema(
            project, reconstructed, required, development_max
        )
        missing = sorted(set(required) - set(reconstructed.columns))
        if missing:
            raise RuntimeError(f"future feature reconstruction missing {len(missing)} features: {missing[:20]}")
        parity = _parity_audit(project, reconstructed, required, development_max, output)

        future = reconstructed[reconstructed["date"] > development_max].copy()
        keep = ["date", "ticker", *required]
        future = future[keep].sort_values(["date", "ticker"]).reset_index(drop=True)
        future_path = output / "FUTURE_SEALED_FEATURES_NO_TARGET.parquet"
        atomic_parquet(future, future_path)

        regime_result = build_market_regimes(
            reconstructed["date"].astype("int64").to_numpy(),
            pd.to_numeric(reconstructed[gate.get("market_proxy_feature", "u_etf_kodex200_ret_1")], errors="coerce").to_numpy(),
            rebound_5d=0.03,
            rebound_drawdown_60=-0.03,
        )
        future_calendar = regime_result.calendar[regime_result.calendar["date"] > development_max].copy()
        atomic_csv(future_calendar, output / "FUTURE_SEALED_REGIME_CALENDAR_NO_TARGET.csv")
        counts = future_calendar["regime"].value_counts().to_dict()
        feature_missing_rate = future[required].replace([np.inf, -np.inf], np.nan).isna().mean()
        quality = pd.DataFrame({
            "feature": required,
            "future_missing_rate": [float(feature_missing_rate[name]) for name in required],
            "future_unique_count": [int(pd.to_numeric(future[name], errors="coerce").nunique(dropna=True)) for name in required],
        })
        atomic_csv(quality, output / "FUTURE_SEALED_FEATURE_QUALITY.csv")
        manifest = {
            "status": "FEATURE_COMPLETE_NO_TARGET_CONSUMED",
            "created_epoch": time.time(), "elapsed_seconds": time.time() - started,
            "gate_hash": gate["artifact_hash"], "development_max": str(development_max.date()),
            "path": str(future_path), "sha256": file_sha256(future_path),
            "rows": len(future), "tickers": int(future["ticker"].nunique()),
            "dates": int(future["date"].nunique()), "date_min": str(future["date"].min().date()),
            "date_max": str(future["date"].max().date()), "required_features": len(required),
            "missing_features": missing, "regime_counts": {key: int(value) for key, value in counts.items()},
            "regimes_present": len(counts), "parity": parity, "base": base_audit,
            "supplemented_reference_features": supplemented,
            "training_schema_alignment": alignment,
            "finance_summary": finance_summary, "target_columns_present": [],
            "target_read": False, "sealed_predictions_made": False,
        }
        atomic_json(manifest, output / "FUTURE_SEALED_FEATURE_MANIFEST.json")
        return manifest
    finally:
        release_lock(lock)


def parse_args() -> argparse.Namespace:
    package = Path(__file__).resolve().parent
    project = package.parent
    parser = argparse.ArgumentParser(description="Reconstruct feature-complete future interval without reading targets")
    parser.add_argument("--output", default=str(project / "crashwatch_ai_data/regime_3d5_submodel_gate_v1"))
    parser.add_argument("--total-threads", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
