from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb


DATE_CANDIDATES = ("date", "trade_date", "trading_date", "datetime", "dt")
TICKER_CANDIDATES = ("ticker", "stock_code", "code", "symbol")


def _rolling_percentile_rank(values: pd.Series, window: int) -> pd.Series:
    array = values.to_numpy(dtype=float)
    output = np.full(len(array), np.nan, dtype=float)
    for index in range(len(array)):
        history = array[max(0, index - window + 1) : index + 1]
        history = history[np.isfinite(history)]
        if len(history) < max(20, min(window, 60)):
            continue
        current = array[index]
        if np.isfinite(current):
            output[index] = float(np.mean(history <= current))
    return pd.Series(output, index=values.index)


def build_regime_calendar(frame: pd.DataFrame, date_column: str, market_feature: str, minimum_history_dates: int) -> pd.DataFrame:
    daily = frame[[date_column, market_feature]].copy()
    daily[market_feature] = pd.to_numeric(daily[market_feature], errors="coerce")
    daily = daily.groupby(date_column, sort=True)[market_feature].median().reset_index(name="ret1")
    returns = daily["ret1"].astype(float)
    gross = 1.0 + returns.fillna(0.0)
    daily["ret_5"] = gross.rolling(5, min_periods=5).apply(np.prod, raw=True) - 1.0
    daily["ret_20"] = gross.rolling(20, min_periods=20).apply(np.prod, raw=True) - 1.0
    daily["vol_20"] = returns.rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
    daily["vol_rank_252"] = _rolling_percentile_rank(daily["vol_20"], 252)
    index_level = gross.cumprod()
    daily["drawdown_60"] = index_level / index_level.rolling(60, min_periods=20).max() - 1.0
    regimes: list[str] = []
    for row in daily.itertuples(index=False):
        r5 = float(row.ret_5) if np.isfinite(row.ret_5) else 0.0
        r20 = float(row.ret_20) if np.isfinite(row.ret_20) else 0.0
        drawdown = float(row.drawdown_60) if np.isfinite(row.drawdown_60) else 0.0
        vol_rank = float(row.vol_rank_252) if np.isfinite(row.vol_rank_252) else 0.5
        high_vol = vol_rank >= 0.70
        if r5 <= -0.05 or r20 <= -0.08:
            regime = "CRASH_STRESS"
        elif r5 >= 0.03 and drawdown <= -0.03:
            regime = "REBOUND"
        elif r20 >= 0.03:
            regime = "BULL_HIGH_VOL" if high_vol else "BULL_LOW_VOL"
        elif r20 <= -0.03:
            regime = "BEAR_HIGH_VOL" if high_vol else "BEAR_LOW_VOL"
        else:
            regime = "SIDEWAYS_HIGH_VOL" if high_vol else "SIDEWAYS_LOW_VOL"
        regimes.append(regime)
    daily["regime"] = regimes
    daily["history_dates"] = np.arange(1, len(daily) + 1, dtype=np.int32)
    daily["history_sufficient"] = daily["history_dates"] >= int(minimum_history_dates)
    daily["market_proxy_available"] = daily["ret1"].notna()
    return daily


def _detect_column(columns: list[str], requested: str, candidates: tuple[str, ...], kind: str) -> str:
    if requested != "AUTO":
        if requested not in columns:
            raise KeyError(f"{kind} column not found: {requested}")
        return requested
    lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise KeyError(f"unable to detect {kind} column; use --{kind}-column")


def _normalize_ticker(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() else text


@dataclass
class DeploymentModel:
    root: Path
    manifest: dict[str, Any]
    features: list[str]
    lightgbm_models: list[lgb.Booster]
    xgboost_models: list[xgb.Booster]

    @classmethod
    def load(cls, root: Path) -> "DeploymentModel":
        root = root.resolve()
        manifest = json.loads((root / "config" / "deployment_manifest.json").read_text(encoding="utf-8"))
        features = json.loads((root / manifest["features"]["path"]).read_text(encoding="utf-8"))
        lightgbm_models: list[lgb.Booster] = []
        xgboost_models: list[xgb.Booster] = []
        for artifact in manifest["artifacts"]:
            path = root / artifact["path"]
            if artifact["family"] == "lightgbm":
                # The Windows LightGBM native loader can corrupt non-ASCII
                # paths. Python reads the file and passes its contents instead.
                lightgbm_models.append(lgb.Booster(model_str=path.read_text(encoding="utf-8")))
            elif artifact["family"] == "xgboost":
                booster = xgb.Booster()
                # bytearray loading is also independent of the install path.
                booster.load_model(bytearray(path.read_bytes()))
                booster.set_param({"device": "cpu", "nthread": max(1, min(16, __import__("os").cpu_count() or 1))})
                xgboost_models.append(booster)
        expected = len(manifest["ensemble"]["seeds"])
        if len(lightgbm_models) != expected or len(xgboost_models) != expected:
            raise RuntimeError(f"model artifact count mismatch LGB={len(lightgbm_models)} XGB={len(xgboost_models)} expected={expected}")
        return cls(root, manifest, features, lightgbm_models, xgboost_models)

    def predict_matrix(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        lgb_prediction = np.vstack([model.predict(matrix) for model in self.lightgbm_models]).mean(axis=0)
        dmatrix = xgb.DMatrix(matrix, feature_names=self.features)
        xgb_prediction = np.vstack([model.predict(dmatrix) for model in self.xgboost_models]).mean(axis=0)
        xgb_weight = float(self.manifest["ensemble"]["xgboost_weight"])
        result = (1.0 - xgb_weight) * lgb_prediction + xgb_weight * xgb_prediction
        if not np.isfinite(result).all():
            raise RuntimeError("model produced non-finite risk scores")
        return np.asarray(result, dtype=np.float64)


def score_frame(
    frame: pd.DataFrame,
    model: DeploymentModel,
    *,
    date_column: str = "AUTO",
    ticker_column: str = "AUTO",
    minimum_history_dates: int | None = None,
    max_row_missing_rate: float = 0.995,
) -> pd.DataFrame:
    data = frame.copy()
    date_column = _detect_column(list(data.columns), date_column, DATE_CANDIDATES, "date")
    ticker_column = _detect_column(list(data.columns), ticker_column, TICKER_CANDIDATES, "ticker")
    missing = [feature for feature in model.features if feature not in data.columns]
    if missing:
        raise KeyError(f"missing required model features ({len(missing)}): {missing[:30]}")
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    if data[date_column].isna().any():
        raise ValueError(f"date parse failed rows={int(data[date_column].isna().sum())}")
    data["__ticker__"] = data[ticker_column].map(_normalize_ticker)
    data["__input_order__"] = np.arange(len(data), dtype=np.int64)
    data.sort_values([date_column, "__ticker__", "__input_order__"], kind="mergesort", inplace=True)
    numeric = data[model.features].apply(pd.to_numeric, errors="coerce")
    matrix = numeric.to_numpy(dtype=np.float32, copy=True)
    matrix[~np.isfinite(matrix)] = np.nan
    missing_rate = np.isnan(matrix).mean(axis=1)

    market_feature = str(model.manifest["inference"]["market_proxy_feature"])
    minimum_history = int(minimum_history_dates or model.manifest["inference"]["minimum_history_dates"])
    calendar = build_regime_calendar(data, date_column, market_feature, minimum_history)
    data = data.merge(
        calendar[[date_column, "regime", "history_dates", "history_sufficient", "market_proxy_available"]],
        on=date_column,
        how="left",
        validate="many_to_one",
    )
    active_regimes = set(model.manifest["inference"]["active_regimes"])
    status = np.full(len(data), "ABSTAIN_ABNORMAL_MARKET", dtype=object)
    status[~data["history_sufficient"].to_numpy(dtype=bool)] = "ABSTAIN_INSUFFICIENT_HISTORY"
    status[~data["market_proxy_available"].to_numpy(dtype=bool)] = "ABSTAIN_MARKET_PROXY_MISSING"
    poor = missing_rate > float(max_row_missing_rate)
    status[poor] = "ABSTAIN_POOR_FEATURE_COVERAGE"
    eligible = (
        data["history_sufficient"].to_numpy(dtype=bool)
        & data["market_proxy_available"].to_numpy(dtype=bool)
        & data["regime"].isin(active_regimes).to_numpy(dtype=bool)
        & ~poor
    )
    status[eligible] = "SCORED_NORMAL_MARKET"
    risk = np.full(len(data), np.nan, dtype=np.float64)
    if eligible.any():
        risk[eligible] = model.predict_matrix(matrix[eligible])

    output = pd.DataFrame(
        {
            "date": data[date_column].to_numpy(),
            "ticker": data["__ticker__"].to_numpy(),
            "regime": data["regime"].to_numpy(),
            "status": status,
            "risk_score": risk,
            "feature_missing_rate": missing_rate,
            "history_dates": data["history_dates"].to_numpy(),
            "model_id": model.manifest["model_id"],
        }
    )
    output["risk_rank"] = np.nan
    output["risk_percentile"] = np.nan
    output["alert_top3pct"] = False
    for _, indexes in output[output["status"].eq("SCORED_NORMAL_MARKET")].groupby("date").groups.items():
        indexes = np.asarray(list(indexes), dtype=np.int64)
        order = indexes[np.argsort(-output.loc[indexes, "risk_score"].to_numpy(), kind="mergesort")]
        count = len(order)
        output.loc[order, "risk_rank"] = np.arange(1, count + 1)
        output.loc[order, "risk_percentile"] = 1.0 - (np.arange(count, dtype=float) / max(1, count))
        alert_count = max(1, int(math.ceil(count * 0.03)))
        output.loc[order[:alert_count], "alert_top3pct"] = True
    return output


def read_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".gz"} or path.name.lower().endswith(".csv.gz"):
        return pd.read_csv(path, low_memory=False, dtype={name: "string" for name in TICKER_CANDIDATES})
    raise ValueError("input must be parquet, csv, or csv.gz")


def main() -> None:
    parser = argparse.ArgumentParser(description="CrashWatch frozen NormalMarket 3D/-5% inference")
    parser.add_argument("--input", required=True, help="feature panel parquet/csv/csv.gz")
    parser.add_argument("--output", required=True, help="prediction parquet or csv")
    parser.add_argument("--date-column", default="AUTO")
    parser.add_argument("--ticker-column", default="AUTO")
    parser.add_argument("--minimum-history-dates", type=int, default=None)
    parser.add_argument("--max-row-missing-rate", type=float, default=0.995)
    parser.add_argument("--latest-only", action="store_true", help="write only the latest input date after using all earlier dates as gate history")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    model = DeploymentModel.load(root)
    source = read_input(Path(args.input).resolve())
    result = score_frame(
        source,
        model,
        date_column=args.date_column,
        ticker_column=args.ticker_column,
        minimum_history_dates=args.minimum_history_dates,
        max_row_missing_rate=args.max_row_missing_rate,
    )
    if args.latest_only and not result.empty:
        result = result[result["date"].eq(result["date"].max())].copy()
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() in {".parquet", ".pq"}:
        result.to_parquet(destination, index=False)
    else:
        result.to_csv(destination, index=False, encoding="utf-8-sig")
    print(json.dumps({"rows": len(result), "scored": int(result["status"].eq("SCORED_NORMAL_MARKET").sum()), "alerts": int(result["alert_top3pct"].sum()), "output": str(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
