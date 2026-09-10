from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import networkx as nx
import pandas as pd

from surge_leadlag_common_v11 import SCHEMA_VERSION, sha256_file


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args(); output=args.output.resolve()
    failures=[]; checks=0
    def check(condition: bool, message: str) -> None:
        nonlocal checks; checks+=1
        if not condition: failures.append(message)
    required=["RUN_STATUS.json","LEADLAG_MANIFEST_V11.json","LEADLAG_RECOMMENDATION_V11.json","ticker_universe.csv",
              "ticker_pair_contemporaneous_map.csv","ticker_pair_direction_agreement.csv","ticker_pair_tail_dependence.csv",
              "lead_lag_cross_correlation_by_fold.csv","lead_lag_cross_correlation_summary.csv","lagged_directional_lift.csv",
              "lagged_tail_event_lift.csv","event_response_curves.csv","event_response_summary.csv","pairwise_granger_diagnostics.csv",
              "lagged_mutual_information.csv","rolling_pair_state.csv","rolling_lead_lag.csv","ticker_pair_divergence_map.csv",
              "ticker_self_leadlag_map.csv","lead_to_surge3d_map.csv","ticker_leader_follower_scores.csv","lead_lag_cascades.csv",
              "directed_network_edges.csv","directed_network_nodes.csv","ticker_lead_lag_network.graphml","ticker_contemporaneous_network.graphml",
              "ticker_contemporaneous_correlation_heatmap.png","ticker_direction_agreement_heatmap.png","ticker_best_lag_heatmap.png",
              "ticker_lead_strength_heatmap.png","ticker_lead_lag_network.png","LEADLAG_MAP_REPORT_KO.md","LEADLAG_FEATURE_MANIFEST_V11.json","OUTPUT_INVENTORY_V11.json"]
    for name in required: check((output/name).is_file() and (output/name).stat().st_size>0,f"missing/empty: {name}")
    if failures:
        print(json.dumps({"status":"FAIL","checks":checks,"failures":failures},indent=2)); raise SystemExit(1)
    status=json.loads((output/"RUN_STATUS.json").read_text(encoding="utf-8")); manifest=json.loads((output/"LEADLAG_MANIFEST_V11.json").read_text(encoding="utf-8")); recommendation=json.loads((output/"LEADLAG_RECOMMENDATION_V11.json").read_text(encoding="utf-8"))
    check(status.get("status")=="SUCCESS","run status not SUCCESS"); check(manifest.get("schema")==SCHEMA_VERSION,"manifest schema"); check(recommendation.get("status")=="LEADLAG_MAP_COMPLETE","recommendation status")
    check(manifest.get("fold_contract")=={"discovery":[0,1,2],"development":[3,4],"confirmation":[5,6],"recent_audit":[7]},"fold isolation")
    check(manifest.get("residual_parameters_train_only") is True,"train-only residuals"); check(manifest.get("rolling_windows_trailing_only") is True,"trailing rolling")
    universe=pd.read_csv(output/"ticker_universe.csv",dtype={"ticker":str}); check(len(universe)==recommendation["tickers"],"ticker count"); check(universe.ticker.str.len().eq(6).all(),"ticker leading zero preservation")
    expected_pairs=len(universe)*(len(universe)-1)//2; check(recommendation["pairs"]==expected_pairs,"pair count")
    lag=pd.read_csv(output/"lead_lag_cross_correlation_by_fold.csv",dtype={"ticker_a":str,"ticker_b":str}); check(lag.lag.between(-5,5).all(),"lag range"); check(set(lag.fold_id.unique())==set(range(8)),"all folds")
    check(set(lag.variant.unique()) >= {"raw_return","cross_demean","market_residual","bucket_residual","return_rank","direction_sign"},"return variants")
    summary=pd.read_csv(output/"lead_lag_cross_correlation_summary.csv",dtype={"ticker_a":str,"ticker_b":str,"leader":str,"follower":str}); check(summary.discovery_q_value.dropna().between(0,1).all(),"FDR bounds")
    strong=summary.loc[summary.strong_directed_edge.astype(bool)]; check((strong.discovery_sign_consistency>=.8).all(),"strong sign consistency"); check((strong.discovery_valid_folds>=2).all(),"strong discovery folds"); check(strong.development_same_direction.astype(bool).all(),"fixed development direction")
    rolling=pd.read_csv(output/"rolling_lead_lag.csv"); check(not rolling.uses_centered_window.astype(bool).any(),"no centered rolling"); check(set(rolling.window.unique())=={60,120,252},"rolling windows")
    granger=pd.read_csv(output/"pairwise_granger_diagnostics.csv"); check(granger.q_value.dropna().between(0,1).all(),"granger FDR"); check(set(granger.lag_order.unique())=={1,2,3,5},"granger lags")
    features=json.loads((output/"LEADLAG_FEATURE_MANIFEST_V11.json").read_text(encoding="utf-8")); check(features.get("future_values_used") is False,"feature future values"); check(all(item.get("available_at_t") is True for item in features.get("features",[])),"feature availability")
    check(len(nx.read_graphml(output/"ticker_lead_lag_network.graphml").nodes)==len(universe),"directed graph nodes"); check(len(nx.read_graphml(output/"ticker_contemporaneous_network.graphml").nodes)==len(universe),"contemporaneous graph nodes")
    for name in ["ticker_contemporaneous_correlation_heatmap.png","ticker_direction_agreement_heatmap.png","ticker_best_lag_heatmap.png","ticker_lead_strength_heatmap.png","ticker_lead_lag_network.png"]: check((output/name).stat().st_size>10_000,f"plot too small: {name}")
    report=(output/"LEADLAG_MAP_REPORT_KO.md").read_text(encoding="utf-8"); check("not proof of economic causality" in report,"causality disclaimer")
    inventory=json.loads((output/"OUTPUT_INVENTORY_V11.json").read_text(encoding="utf-8")); inventory_map={item["file"]:item for item in inventory}
    for name,item in inventory_map.items(): check((output/name).stat().st_size==item["bytes"] and sha256_file(output/name)==item["sha256"],f"inventory mismatch: {name}")
    if recommendation.get("probe_run"):
        for name in ["leadlag_probe_predictions.csv","leadlag_probe_metrics.csv","leadlag_probe_champions.csv","leadlag_probe_portfolio_by_fold.csv"]: check((output/name).is_file(),f"probe missing: {name}")
        portfolio=pd.read_csv(output/"leadlag_probe_portfolio_by_fold.csv"); check(set(portfolio.fold_id.unique())=={3,4,5,6,7},"probe fold coverage")
    result={"status":"PASS" if not failures else "FAIL","checks":checks,"failures":failures,"tickers":len(universe),"pairs":expected_pairs,"strong_edges":int(recommendation["strong_directed_edges"])}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if failures: raise SystemExit(1)


if __name__=="__main__": main()
