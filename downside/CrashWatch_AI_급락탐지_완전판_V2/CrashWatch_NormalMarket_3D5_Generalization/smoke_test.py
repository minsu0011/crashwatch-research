from __future__ import annotations
import json, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from cwregime.target import build_3d5_target_from_ret1
from cwregime.regimes import build_market_regimes
from cwnormal.gate import build_normal_gate, NORMAL_REGIMES

root=Path(__file__).resolve().parent
cfg=json.load((root/'config_normal_market_3d5.json').open(encoding='utf-8'))
assert cfg['target']['drop_threshold']==-0.05
assert cfg['target']['horizon_trading_days']==3
manifest=json.load((root/'seed_results'/'profile_manifest.json').open(encoding='utf-8'))
assert len(manifest['profiles']['P2_DEDUP_CLEAN']['features'])==371
assert len(manifest['profiles']['P7_CORR095_PLUS_CONDITIONAL']['features'])==341
# 48 tickers x 700 dates synthetic, date-major.
rng=np.random.default_rng(17); nd=700; nt=48
dates_unique=pd.bdate_range('2020-01-02', periods=nd).astype('int64').to_numpy()
dates=np.repeat(dates_unique,nt); tickers=np.tile(np.array([f'{i:06d}' for i in range(nt)]),nd)
market=rng.normal(0.0002,0.008,size=nd); market_rows=np.repeat(market,nt)
idio=rng.normal(0,0.018,size=len(dates)); ret=market_rows+idio
# force a few 3-day -5% events
for d in [200,350,500]: ret[(d+1)*nt:(d+1)*nt+5]=-0.03; ret[(d+2)*nt:(d+2)*nt+5]=-0.03
t=build_3d5_target_from_ret1(dates,tickers,ret,horizon_days=3,drop_threshold=-0.05)
assert t.audit['drop_threshold']==-0.05 and t.valid.sum()>0 and t.label.sum()>0
r=build_market_regimes(dates,market_rows)
g=build_normal_gate(dates,r.calendar)
assert set(NORMAL_REGIMES)=={'BULL_LOW_VOL','SIDEWAYS_LOW_VOL','BEAR_LOW_VOL'}
assert 0 < g.audit['date_coverage'] < 1
print('SMOKE_PASS')
print('target positives=',int(t.label.sum()),'normal coverage=',g.audit['date_coverage'])
