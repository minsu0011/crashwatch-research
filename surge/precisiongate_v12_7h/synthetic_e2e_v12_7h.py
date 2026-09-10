from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from test_surge_precision_gate_v12_7h import make_meta
from surge_precision_gate_v12_7h import (
    TrialConfig, build_prediction_frame, config_for_family, dev_summary,
    fit_trial_model, predict_trial_model, rank_hard_fp_features,
)

OUT=Path("synthetic_output_v12_7h")
OUT.mkdir(exist_ok=True)
d,e=make_meta(33)
rank=rank_hard_fp_features(d,["raw_good","raw_bad","raw_inv"],candidate_quantile=0.4)
rank.to_csv(OUT/"hard_fp_feature_ranking.csv",index=False)
cfg=config_for_family(1,42)
cfg=TrialConfig(**{**vars(cfg),"backend":"logit","candidate_quantile":0.4,"feature_k":2,"alpha":0.75})
selected=rank.head(2).feature.tolist()
preds=[]
for seed in [11,23,37]:
    model=fit_trial_model(d,selected,cfg,seed=seed,cpu_threads=2,gpu_available=False)
    gate=predict_trial_model(model,e)
    preds.append(build_prediction_frame(e,gate,cfg,minimum_evidence_count=1))
mean_gate=np.mean(np.vstack([x.gate_prob.to_numpy() for x in preds]),axis=0)
from surge_precision_gate_v12_7h import ensemble_prediction
ens=ensemble_prediction(preds,cfg)
summary=dev_summary(ens,dev_folds=[3,4],minimum_alerts=10,target_precision=0.5)
ens.to_csv(OUT/"synthetic_predictions.csv",index=False)
(OUT/"SYNTHETIC_RESULT.json").write_text(json.dumps({"status":"PASS","selected":selected,"summary":summary},indent=2,default=str),encoding="utf-8")
print("SYNTHETIC_E2E_PASS", selected, summary["dev_safe"], summary["mean_dev_pr_auc_delta"])
