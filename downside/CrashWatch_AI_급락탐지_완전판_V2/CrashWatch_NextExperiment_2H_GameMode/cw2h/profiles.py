from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from cw7h.utils import hash_strings, atomic_json

HARMFUL_THREE=["t_finshort_balance_slope_20","t_finshort_volume_sum_5","t_taildep_cocrash_freq_120"]
PROTECTED_HARMFUL_TEN=[
"t_event_other_count_20","t_finshort_balance_slope_20","t_finshort_volume_sum_20","t_finshort_volume_sum_5",
"t_lending_balance_z_60","t_price_ret_20","t_taildep_cocrash_freq_120","u_lending_balance_mean_change5",
"u_tailnet_corr_q90_60","u_tailnet_largest_eigen_share_60"]

def build_profiles(package_root:Path,feature_names:list[str],output_dir:Path):
    m=json.load(open(package_root/'seed_results/profile_manifest.json',encoding='utf-8'))
    available=set(feature_names)
    base={k:[f for f in v['features'] if f in available] for k,v in m['profiles'].items()}
    p1=base['P1_EXACT_DEDUP']; p2=base['P2_DEDUP_CLEAN']; p7=base['P7_CORR095_PLUS_CONDITIONAL']
    high=[f for f in m['high_missing_drop'] if f in available]
    harm=[f for f in HARMFUL_THREE if f in available]
    def drop(features,names):
        s=set(names); return [f for f in features if f not in s]
    def add(features,names):
        s=set(features); return list(features)+[f for f in names if f not in s]
    named={
      'C0_P1_EXACT_DEDUP':p1,
      'C1_P1_MINUS_HIGH_MISSING':drop(p1,high),
      'C2_P1_MINUS_HARMFUL3':drop(p1,harm),
      'C3_P2_ALL5_REMOVED':p2,
      'C4_P2_RESTORE_HARMFUL3':drop(p1,high),
      'C5_P2_RESTORE_BALANCE_SLOPE':add(p2,[harm[0]]) if len(harm)>0 else p2,
      'C6_P2_RESTORE_VOLUME_SUM5':add(p2,[harm[1]]) if len(harm)>1 else p2,
      'C7_P2_RESTORE_COCRASH_FREQ':add(p2,[harm[2]]) if len(harm)>2 else p2,
      'P7_CORR095_PLUS_CONDITIONAL':p7,
      'P0_FULL_439':base['P0_FULL_439'],
      'P2_DEDUP_CLEAN':p2,
    }
    # C1 and C4 are mathematically the same set. Train once, report both aliases.
    aliases={}
    unique={}
    hash_to_name={}
    for name,features in named.items():
        h=hash_strings(features)
        if h in hash_to_name: aliases[name]=hash_to_name[h]
        else: hash_to_name[h]=name; unique[name]=features
    required_lgb=['C0_P1_EXACT_DEDUP','C1_P1_MINUS_HIGH_MISSING','C2_P1_MINUS_HARMFUL3','C3_P2_ALL5_REMOVED','C5_P2_RESTORE_BALANCE_SLOPE','C6_P2_RESTORE_VOLUME_SUM5','C7_P2_RESTORE_COCRASH_FREQ','P7_CORR095_PLUS_CONDITIONAL']
    lgb={name:named[name] for name in required_lgb}
    xgb={name:named[name] for name in ['P0_FULL_439','P2_DEDUP_CLEAN','P7_CORR095_PLUS_CONDITIONAL']}
    manifest={
      'profiles':{k:{'count':len(v),'feature_hash':hash_strings(v),'features':v} for k,v in named.items()},
      'aliases':aliases,'lgb_profiles':list(lgb),'xgb_profiles':list(xgb),
      'high_missing_two':high,'harmful_three':harm,'protected_harmful_ten':[f for f in PROTECTED_HARMFUL_TEN if f in available],
      'warning':'feature_master_decision DROP labels are not used.'
    }
    atomic_json(manifest,output_dir/'experiment_profile_manifest.json')
    return named,lgb,xgb,aliases,manifest
