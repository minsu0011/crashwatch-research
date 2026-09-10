from __future__ import annotations
import argparse,json,math
from pathlib import Path
import pandas as pd

REQUIRED=[
 "RUN_STATUS.json","GPU_PREFLIGHT.json","DATA_AUDIT.json","search_family_results.csv",
 "TOP_SEARCH_FAMILIES.csv","robust_family_results.csv","CHAMPION_CONFIG_V12_7H.json",
 "v12_7h_champion_predictions.csv","v12_7h_metrics_by_fold.csv","v12_7h_frozen_policy_by_fold.csv",
 "FINAL_RECOMMENDATION_V12_7H.json","LEAKAGE_CONTRACT_V12_7H.json",
]

def main():
 p=argparse.ArgumentParser(); p.add_argument("--output",required=True); a=p.parse_args(); root=Path(a.output)
 checks=[]
 for name in REQUIRED:
  checks.append((f"exists:{name}",(root/name).exists()))
 status=json.loads((root/"RUN_STATUS.json").read_text(encoding="utf-8")) if (root/"RUN_STATUS.json").exists() else {}
 checks.append(("run_success",status.get("status")=="SUCCESS"))
 final=json.loads((root/"FINAL_RECOMMENDATION_V12_7H.json").read_text(encoding="utf-8")) if (root/"FINAL_RECOMMENDATION_V12_7H.json").exists() else {}
 checks.append(("parent_v10_2",final.get("direct_parent")=="V10.2"))
 checks.append(("v11_reference_only",final.get("v11_v11_1")=="REFERENCE_ONLY_NOT_USED"))
 checks.append(("no_production_auto_enable",str(final.get("production_action","")).startswith("NO_ALERT")))
 if (root/"v12_7h_frozen_policy_by_fold.csv").exists():
  pol=pd.read_csv(root/"v12_7h_frozen_policy_by_fold.csv")
  checks.append(("folds_3_to_7",set(pol.fold_id.astype(int))=={3,4,5,6,7}))
  checks.append(("one_row_per_fold",len(pol)==5))
 if (root/"search_family_results.csv").exists():
  s=pd.read_csv(root/"search_family_results.csv")
  checks.append(("search_nonempty",len(s)>0))
 if (root/"v12_7h_champion_predictions.csv").exists():
  p0=pd.read_csv(root/"v12_7h_champion_predictions.csv")
  checks.append(("prediction_required_cols",{"ticker","fold_id","target","base_score_raw","gate_prob","score","candidate_eligible"}.issubset(p0.columns)))
 result={"checks":len(checks),"passed":sum(bool(v) for _,v in checks),"failed":[k for k,v in checks if not v],"details":checks}
 (root/"VERIFICATION_RESULTS_V12_7H.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
 print(json.dumps(result,indent=2))
 if result["failed"]: raise SystemExit(1)
if __name__=="__main__": main()
