from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from surge_precision_gate_v12 import (
    apply_frozen_threshold, build_discovery_separator_specs, combine_base_and_gate,
    evaluate_scores, fit_evidence_transformer, fit_gate_bundle, predict_gate_bundle,
    select_frozen_dev_threshold, serialize_specs, transform_evidence,
)


def main() -> None:
    root = Path(__file__).resolve().parent / "synthetic_output_v12"
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260815)
    tickers = ["005930","000660","042700","079550","086520"]
    rows=[]
    rid=0
    for fold in range(8):
        for ti,t in enumerate(tickers):
            for i in range(60):
                y=int(rng.random() < (0.23+0.02*(ti%2)))
                base=0.12+0.48*y+rng.normal(0,0.11)
                if y==0 and rng.random()<0.28:
                    base+=0.60
                base=float(np.clip(base,0.01,0.99))
                sep1=(2.5 if y else -1.1)+rng.normal(0,0.70)
                sep2=(-1.9 if y else 0.9)+rng.normal(0,0.75)
                rows.append({"row_index":rid,"source_row_id":rid+100000,"ticker":t,"fold_id":fold,"target":y,"base_score_raw":base,"sep1":sep1,"sep2":sep2})
                rid+=1
    df=pd.DataFrame(rows)
    dmap=[]
    for t in tickers:
        dmap += [
            {"ticker":t,"axis":"AB","node_id":"sep1","source_feature":"sep1","selection_direction":1,"precision_separator_score_v10_2":2.2,"precision_separator_selection_candidate":True},
            {"ticker":t,"axis":"AB","node_id":"sep2","source_feature":"sep2","selection_direction":-1,"precision_separator_score_v10_2":1.8,"precision_separator_selection_candidate":True},
        ]
    specs=build_discovery_separator_specs(pd.DataFrame(dmap),df.columns,max_features_per_ticker=2)
    discovery=df[df.fold_id.isin([0,1,2])].copy()
    evaluation=df[df.fold_id.isin([3,4,5,6,7])].copy()
    tr=fit_evidence_transformer(discovery,specs,max_slots=2)
    dm=pd.concat([discovery.reset_index(drop=True),transform_evidence(discovery,tr).reset_index(drop=True)],axis=1)
    em=pd.concat([evaluation.reset_index(drop=True),transform_evidence(evaluation,tr).reset_index(drop=True)],axis=1)
    bundle=fit_gate_bundle(dm,backend="logit",candidate_quantile=0.50,max_slots=2,min_ticker_rows=25,min_ticker_class=4)
    gate=predict_gate_bundle(bundle,em)
    pred=em[["row_index","source_row_id","ticker","fold_id","target","base_score_raw"]].copy()
    pred["base_hist_rank"]=em["base_hist_rank"].to_numpy(float)
    pred["ab_evidence_count"]=em["ab_evidence_count"].to_numpy(float)
    pred["candidate_eligible"]=gate.candidate_eligible.to_numpy(bool)
    pred["gate_prob"]=gate.gate_prob.to_numpy(float)
    pred["v12_score"]=combine_base_and_gate(pred.base_score_raw,pred.gate_prob,1.0)
    policy=select_frozen_dev_threshold(pred,minimum_alerts=30,target_precision=0.70)
    metrics=evaluate_scores(pred,score_column="v12_score",minimum_alerts=30)
    baseline=pred.copy(); baseline["base_eval_score"]=baseline.base_score_raw
    baseline_metrics=evaluate_scores(baseline,score_column="base_eval_score",minimum_alerts=30)
    frozen=apply_frozen_threshold(pred,policy["threshold"],folds=[3,4,5,6,7])
    frozen["role"]=frozen.fold_id.map({3:"development",4:"development",5:"confirmation",6:"confirmation",7:"recent_diagnostic"})
    frozen["gate_pass"]=(frozen.alerts>=30)&(frozen.precision>=0.70)
    ticker=[]
    for (t,f),g in pred.groupby(["ticker","fold_id"]):
        ticker.append({"ticker":t,"fold_id":f,"rows":len(g),"positives":int(g.target.sum()),"eligible_rows":int(g.candidate_eligible.sum())})
    manifest=pd.DataFrame(serialize_specs(specs))
    manifest.to_csv(root/"V12_DISCOVERY_SEPARATOR_MANIFEST.csv",index=False)
    pd.DataFrame([{"config_id":"logit__q0.50__a1.00","backend":"logit","candidate_quantile":0.5,"alpha":1.0,"dev_safe":policy["safe"],"frozen_threshold":policy["threshold"],"min_dev_precision":policy.get("min_dev_precision"),"mean_dev_precision":policy.get("mean_dev_precision")}]).to_csv(root/"v12_config_search.csv",index=False)
    pred.to_csv(root/"v12_champion_predictions.csv",index=False)
    metrics.to_csv(root/"v12_metrics_by_fold.csv",index=False)
    frozen.to_csv(root/"v12_frozen_policy_by_fold.csv",index=False)
    baseline_metrics.to_csv(root/"v10_2_baseline_matched_by_fold.csv",index=False)
    pd.DataFrame(ticker).to_csv(root/"v12_ticker_metrics_by_fold.csv",index=False)
    contract={
        "direct_parent":"V10.2","v11_v11_1_status":"REFERENCE_ONLY_NOT_USED_AS_MODEL_INPUT",
        "discovery_model_fit_folds":[0,1,2],"development_selection_folds":[3,4],"frozen_evaluation_folds":[5,6,7],
        "base_rank_rule":"V10.2 fold-wide base_rank is forbidden; discovery-history empirical rank only."
    }
    (root/"LEAKAGE_CONTRACT_V12.json").write_text(json.dumps(contract,indent=2),encoding="utf-8")
    final={
        "schema":"crashwatch_surge_precision_gate_v12","status":"V12_COMPLETE","parent":"V10.2","v11_status":"REFERENCE_ONLY",
        "champion":{"config_id":"logit__q0.50__a1.00","dev_safe":bool(policy["safe"]),"frozen_threshold":policy["threshold"]},
        "target_policy":{"precision":0.70,"minimum_alerts":30},"production_action":"NO_ALERT_UNTIL_NEW_FUTURE_DATA",
        "forward_mean_precision":float(frozen.loc[frozen.fold_id>=5,"precision"].mean())
    }
    (root/"FINAL_RECOMMENDATION_V12.json").write_text(json.dumps(final,indent=2),encoding="utf-8")
    (root/"RUN_STATUS.json").write_text(json.dumps({"status":"SUCCESS"},indent=2),encoding="utf-8")
    print(json.dumps({"status":"SUCCESS","dev_safe":bool(policy["safe"]),"threshold":policy["threshold"],"forward_mean_precision":final["forward_mean_precision"]},indent=2))


if __name__=="__main__":
    main()
