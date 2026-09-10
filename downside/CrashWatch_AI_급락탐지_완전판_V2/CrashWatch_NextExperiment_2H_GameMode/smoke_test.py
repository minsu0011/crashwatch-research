from pathlib import Path
import json, tempfile
from cw2h.profiles import build_profiles
root=Path(__file__).resolve().parent
features=[x.strip() for x in open(root/'reference/valid_feature_audit.csv',encoding='utf-8-sig').read().splitlines()[1:] if x.strip()]
# actual CSV parsing
import pandas as pd
f=pd.read_csv(root/'reference/valid_feature_audit.csv')['feature'].astype(str).tolist()
with tempfile.TemporaryDirectory() as d:
 named,lgb,xgb,aliases,m=build_profiles(root,f,Path(d))
 assert len(named['P0_FULL_439'])==439
 assert len(named['P2_DEDUP_CLEAN'])==371
 assert len(named['P7_CORR095_PLUS_CONDITIONAL'])==341
 assert aliases.get('C4_P2_RESTORE_HARMFUL3')=='C1_P1_MINUS_HIGH_MISSING'
 print('SMOKE PASS', {k:len(v) for k,v in lgb.items()}, {k:len(v) for k,v in xgb.items()}, aliases)
