from __future__ import annotations

import argparse
import json
from pathlib import Path
import math
import pandas as pd

REQUIRED = [
    "RUN_STATUS.json",
    "FINAL_RECOMMENDATION_V12.json",
    "LEAKAGE_CONTRACT_V12.json",
    "V12_DISCOVERY_SEPARATOR_MANIFEST.csv",
    "v12_config_search.csv",
    "v12_champion_predictions.csv",
    "v12_metrics_by_fold.csv",
    "v12_frozen_policy_by_fold.csv",
    "v10_2_baseline_matched_by_fold.csv",
    "v12_ticker_metrics_by_fold.csv",
]


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--minimum-alerts",type=int,default=30)
    p.add_argument("--target-precision",type=float,default=0.70)
    args=p.parse_args()
    root=Path(args.output)
    checks=[]
    def check(name, cond, detail=""):
        checks.append({"check":name,"pass":bool(cond),"detail":str(detail)})

    for name in REQUIRED:
        check(f"file:{name}",(root/name).exists(),name)
    if not all((root/x).exists() for x in REQUIRED):
        out={"status":"FAIL","checks":checks}
        (root/"VERIFICATION_V12.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
        print(json.dumps(out,indent=2)); raise SystemExit(1)

    status=json.loads((root/"RUN_STATUS.json").read_text(encoding="utf-8"))
    final=json.loads((root/"FINAL_RECOMMENDATION_V12.json").read_text(encoding="utf-8"))
    contract=json.loads((root/"LEAKAGE_CONTRACT_V12.json").read_text(encoding="utf-8"))
    manifest=pd.read_csv(root/"V12_DISCOVERY_SEPARATOR_MANIFEST.csv",dtype={"ticker":str})
    configs=pd.read_csv(root/"v12_config_search.csv")
    pred=pd.read_csv(root/"v12_champion_predictions.csv",dtype={"ticker":str})
    metrics=pd.read_csv(root/"v12_metrics_by_fold.csv")
    frozen=pd.read_csv(root/"v12_frozen_policy_by_fold.csv")
    base=pd.read_csv(root/"v10_2_baseline_matched_by_fold.csv")

    check("status_success",status.get("status")=="SUCCESS",status.get("status"))
    check("schema",final.get("schema")=="crashwatch_surge_precision_gate_v12",final.get("schema"))
    check("parent_v10_2",final.get("parent")=="V10.2",final.get("parent"))
    check("v11_reference_only",final.get("v11_status")=="REFERENCE_ONLY",final.get("v11_status"))
    check("production_no_alert",str(final.get("production_action","")).startswith("NO_ALERT"),final.get("production_action"))
    check("contract_lineage",contract.get("direct_parent")=="V10.2",contract.get("direct_parent"))
    check("contract_v11_not_input",contract.get("v11_v11_1_status")=="REFERENCE_ONLY_NOT_USED_AS_MODEL_INPUT",contract.get("v11_v11_1_status"))
    check("discovery_folds",contract.get("discovery_model_fit_folds")==[0,1,2],contract.get("discovery_model_fit_folds"))
    check("development_folds",contract.get("development_selection_folds")==[3,4],contract.get("development_selection_folds"))
    check("frozen_folds",contract.get("frozen_evaluation_folds")==[5,6,7],contract.get("frozen_evaluation_folds"))
    check("no_v10_foldwide_base_rank", "forbidden" in str(contract.get("base_rank_rule","")).lower(),contract.get("base_rank_rule"))
    check("manifest_nonempty",len(manifest)>0,len(manifest))
    check("manifest_direction",set(pd.to_numeric(manifest.direction,errors="coerce").dropna().astype(int)).issubset({-1,1}),set(manifest.direction))
    check("manifest_unique_source_per_ticker",not manifest.duplicated(["ticker","source_feature"]).any(),int(manifest.duplicated(["ticker","source_feature"]).sum()))
    check("config_nonempty",len(configs)>0,len(configs))
    check("prediction_unique",not pred.duplicated(["row_index","fold_id"]).any(),int(pred.duplicated(["row_index","fold_id"]).sum()))
    check("prediction_folds",set(pred.fold_id.astype(int).unique())=={3,4,5,6,7},sorted(pred.fold_id.unique().tolist()))
    check("has_hist_rank","base_hist_rank" in pred.columns,pred.columns.tolist())
    check("does_not_use_v10_base_rank","base_rank" not in pred.columns,pred.columns.tolist())
    check("metrics_folds",set(metrics.fold_id.astype(int).unique())=={3,4,5,6,7},sorted(metrics.fold_id.unique().tolist()))
    check("baseline_matched_rows",dict(zip(metrics.fold_id.astype(int),metrics.rows.astype(int)))==dict(zip(base.fold_id.astype(int),base.rows.astype(int))),"row-count comparison")
    check("frozen_folds_output",set(frozen.fold_id.astype(int).unique())=={3,4,5,6,7},sorted(frozen.fold_id.unique().tolist()))
    target=float(final.get("target_policy",{}).get("precision",args.target_precision))
    minimum=int(final.get("target_policy",{}).get("minimum_alerts",args.minimum_alerts))
    dev=frozen[frozen.fold_id.isin([3,4])]
    dev_pass=((dev.alerts>=minimum)&(dev.precision>=target)).all() if len(dev)==2 else False
    champion_safe=bool(final.get("champion",{}).get("dev_safe",False))
    check("dev_safe_consistency",champion_safe==bool(dev_pass),(champion_safe,dev_pass))
    check("threshold_finite_if_safe",(not champion_safe) or math.isfinite(float(final.get("champion",{}).get("frozen_threshold",float("inf")))),final.get("champion",{}).get("frozen_threshold"))
    check("no_false_production_upgrade",not (champion_safe and final.get("production_action")=="ENABLE_ALERT"),final.get("production_action"))
    check("v11_named_files_absent",not any("v11" in p.name.lower() for p in root.iterdir()),[p.name for p in root.iterdir() if "v11" in p.name.lower()])

    ok=all(x["pass"] for x in checks)
    out={"schema":"crashwatch_surge_precision_gate_v12_verifier","status":"PASS" if ok else "FAIL","passed":sum(x["pass"] for x in checks),"total":len(checks),"checks":checks}
    (root/"VERIFICATION_V12.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out,ensure_ascii=False,indent=2))
    raise SystemExit(0 if ok else 1)

if __name__=="__main__":
    main()
