from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Any

from cw7h.utils import atomic_json, hash_strings

FACTOR_FEATURES = [
    "u_finmarket_nfci_change5",
    "u_finmarket_stl_fsi_change5",
    "t_finshort_balance_slope_20",
    "t_finshort_volume_sum_5",
    "t_taildep_cocrash_freq_120",
]
HIGH_MISSING_TWO = FACTOR_FEATURES[:2]
HARMFUL_THREE = FACTOR_FEATURES[2:]
PROTECTED_HARMFUL_TEN = [
    "t_event_other_count_20",
    "t_finshort_balance_slope_20",
    "t_finshort_volume_sum_20",
    "t_finshort_volume_sum_5",
    "t_lending_balance_z_60",
    "t_price_ret_20",
    "t_taildep_cocrash_freq_120",
    "u_lending_balance_mean_change5",
    "u_tailnet_corr_q90_60",
    "u_tailnet_largest_eigen_share_60",
]
PRIORITY_FEATURES = [
    "t_taildep_tail_beta_120",
    "t_taildep_down_corr_60",
    "t_finshort_balance_z_20",
]


def _drop(features: list[str], names: list[str] | set[str]) -> list[str]:
    blocked = set(names)
    return [feature for feature in features if feature not in blocked]


def _add(features: list[str], names: list[str]) -> list[str]:
    result = list(features)
    seen = set(result)
    for feature in names:
        if feature not in seen:
            result.append(feature)
            seen.add(feature)
    return result


def _slug(feature: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", feature).strip("_")


def _factor_name(bits: tuple[int, ...]) -> str:
    return "F_REMOVE_" + "".join(str(int(bit)) for bit in bits)


def build_experiment_profiles(package_root: Path, feature_names: list[str], output_dir: Path) -> dict[str, Any]:
    seed_manifest_path = package_root / "seed_results" / "profile_manifest.json"
    seed_manifest = json.load(seed_manifest_path.open(encoding="utf-8"))
    available = set(feature_names)
    missing_required = [feature for feature in FACTOR_FEATURES + PRIORITY_FEATURES if feature not in available]
    if missing_required:
        raise RuntimeError(f"필수 피처 누락: {missing_required}")

    base_profiles = {
        name: [feature for feature in payload["features"] if feature in available]
        for name, payload in seed_manifest["profiles"].items()
    }
    required_counts = {
        "P0_FULL_439": 439,
        "P1_EXACT_DEDUP": 376,
        "P2_DEDUP_CLEAN": 371,
        "P4_CORR095": 333,
        "P7_CORR095_PLUS_CONDITIONAL": 341,
    }
    for name, count in required_counts.items():
        actual = len(base_profiles.get(name, []))
        if actual != count:
            raise RuntimeError(f"{name} 피처 수 불일치: expected={count}, actual={actual}")

    p0 = base_profiles["P0_FULL_439"]
    p1 = base_profiles["P1_EXACT_DEDUP"]
    p2 = base_profiles["P2_DEDUP_CLEAN"]
    p4 = base_profiles["P4_CORR095"]
    p7 = base_profiles["P7_CORR095_PLUS_CONDITIONAL"]
    conditional_features = [
        feature for feature in seed_manifest.get("conditional_nonrep_added", []) if feature in available
    ]
    if len(conditional_features) != 8:
        raise RuntimeError(f"P7 조건부 추가 피처는 8개여야 합니다: {conditional_features}")

    factorial: dict[str, list[str]] = {}
    factor_bits: dict[str, list[int]] = {}
    bits_to_name: dict[tuple[int, ...], str] = {}
    for bits in itertools.product([0, 1], repeat=len(FACTOR_FEATURES)):
        name = _factor_name(bits)
        removed = [feature for bit, feature in zip(bits, FACTOR_FEATURES) if bit]
        factorial[name] = _drop(p1, removed)
        factor_bits[name] = list(bits)
        bits_to_name[bits] = name

    named_aliases = {
        "C0_P1_EXACT_DEDUP": bits_to_name[(0, 0, 0, 0, 0)],
        "C1_P1_MINUS_HIGH_MISSING": bits_to_name[(1, 1, 0, 0, 0)],
        "C2_P1_MINUS_HARMFUL3": bits_to_name[(0, 0, 1, 1, 1)],
        "C3_P2_ALL5_REMOVED": bits_to_name[(1, 1, 1, 1, 1)],
        "C4_P2_RESTORE_HARMFUL3": bits_to_name[(1, 1, 0, 0, 0)],
        "C5_P2_RESTORE_BALANCE_SLOPE": bits_to_name[(1, 1, 0, 1, 1)],
        "C6_P2_RESTORE_VOLUME_SUM5": bits_to_name[(1, 1, 1, 0, 1)],
        "C7_P2_RESTORE_COCRASH_FREQ": bits_to_name[(1, 1, 1, 1, 0)],
        "P1_EXACT_DEDUP": bits_to_name[(0, 0, 0, 0, 0)],
        "P2_DEDUP_CLEAN": bits_to_name[(1, 1, 1, 1, 1)],
    }
    if hash_strings(factorial[named_aliases["P2_DEDUP_CLEAN"]]) != hash_strings(p2):
        raise RuntimeError("factorial P2와 seed P2의 피처 집합/순서가 일치하지 않습니다.")

    core_operational = dict(factorial)
    core_operational["P0_FULL_439"] = p0
    core_operational["P4_CORR095"] = p4
    core_operational["P7_CORR095_PLUS_CONDITIONAL"] = p7

    targeted_operational: dict[str, list[str]] = {}
    experiment_meta: dict[str, dict[str, Any]] = {}

    protected = [feature for feature in PROTECTED_HARMFUL_TEN if feature in available]
    for feature in protected:
        slug = _slug(feature)
        # P1: every protected feature is expected to be present. Removal tests the old DROP claim directly.
        name = f"HARMFUL_P1_DROP__{slug}"
        targeted_operational[name] = _drop(p1, [feature])
        experiment_meta[name] = {"family": "harmful10", "context": "P1", "feature": feature, "operation": "drop"}

        # P2 and P7 are tested in their actual operational context: drop if present, add if absent.
        for context, baseline in [("P2", p2), ("P7", p7)]:
            present = feature in baseline
            operation = "drop" if present else "add"
            name = f"HARMFUL_{context}_{operation.upper()}__{slug}"
            targeted_operational[name] = _drop(baseline, [feature]) if present else _add(baseline, [feature])
            experiment_meta[name] = {
                "family": "harmful10", "context": context, "feature": feature, "operation": operation,
            }

    for feature in PRIORITY_FEATURES:
        slug = _slug(feature)
        for context, baseline in [("P2", p2), ("P7", p7)]:
            if feature not in baseline:
                raise RuntimeError(f"우선 유지 피처가 {context}에 없습니다: {feature}")
            name = f"PRIORITY_{context}_DROP__{slug}"
            targeted_operational[name] = _drop(baseline, [feature])
            experiment_meta[name] = {
                "family": "priority", "context": context, "feature": feature, "operation": "drop",
            }

    for feature in conditional_features:
        slug = _slug(feature)
        add_name = f"CONDITIONAL_P4_ADD__{slug}"
        targeted_operational[add_name] = _add(p4, [feature])
        experiment_meta[add_name] = {
            "family": "conditional", "context": "P4", "feature": feature, "operation": "add",
        }
        if feature not in p7:
            raise RuntimeError(f"P7 조건부 피처 누락: {feature}")
        drop_name = f"CONDITIONAL_P7_DROP__{slug}"
        targeted_operational[drop_name] = _drop(p7, [feature])
        experiment_meta[drop_name] = {
            "family": "conditional", "context": "P7", "feature": feature, "operation": "drop",
        }

    # Full-column diagnostic repeats the complete 2^5 factorial and all P1 harmful removals.
    diagnostic_profiles = dict(factorial)
    for feature in protected:
        name = f"HARMFUL_P1_DROP__{_slug(feature)}"
        diagnostic_profiles[name] = _drop(p1, [feature])

    # Deduplicate equal feature sets within each stage, while preserving aliases for reporting.
    def deduplicate(profiles: dict[str, list[str]]) -> tuple[dict[str, list[str]], dict[str, str]]:
        unique: dict[str, list[str]] = {}
        aliases: dict[str, str] = {}
        hash_to_name: dict[str, str] = {}
        for name, features in profiles.items():
            digest = hash_strings(features)
            if digest in hash_to_name:
                aliases[name] = hash_to_name[digest]
            else:
                hash_to_name[digest] = name
                unique[name] = features
        return unique, aliases

    core_unique, core_aliases = deduplicate(core_operational)
    target_unique, target_aliases = deduplicate(targeted_operational)
    diagnostic_unique, diagnostic_aliases = deduplicate(diagnostic_profiles)

    # Named aliases always resolve into the operational core factorial.
    operational_aliases = dict(core_aliases)
    operational_aliases.update(named_aliases)
    operational_aliases.update(target_aliases)

    manifest = {
        "schema_version": "crashwatch_final_selection_profiles_v1",
        "policy": {
            "feature_master_decision_drop_labels_used": False,
            "protected_harmful_ten_are_never_auto_deleted": True,
            "sealed_data_used_for_selection": False,
        },
        "base_profiles": {
            name: {"count": len(features), "feature_hash": hash_strings(features), "features": features}
            for name, features in {
                "P0_FULL_439": p0,
                "P1_EXACT_DEDUP": p1,
                "P2_DEDUP_CLEAN": p2,
                "P4_CORR095": p4,
                "P7_CORR095_PLUS_CONDITIONAL": p7,
            }.items()
        },
        "factor_features": FACTOR_FEATURES,
        "factor_bits": factor_bits,
        "named_aliases": named_aliases,
        "protected_harmful_ten": protected,
        "priority_features": PRIORITY_FEATURES,
        "conditional_features": conditional_features,
        "core_operational": {
            name: {"count": len(features), "feature_hash": hash_strings(features), "features": features}
            for name, features in core_unique.items()
        },
        "targeted_operational": {
            name: {"count": len(features), "feature_hash": hash_strings(features), "features": features}
            for name, features in target_unique.items()
        },
        "diagnostic_full_column": {
            name: {"count": len(features), "feature_hash": hash_strings(features), "features": features}
            for name, features in diagnostic_unique.items()
        },
        "operational_aliases": operational_aliases,
        "diagnostic_aliases": diagnostic_aliases,
        "experiment_meta": experiment_meta,
        "xgboost_profiles": {
            "P0_FULL_439": p0,
            "P2_DEDUP_CLEAN": p2,
            "P7_CORR095_PLUS_CONDITIONAL": p7,
        },
    }
    atomic_json(manifest, output_dir / "experiment_profile_manifest.json")
    return manifest


def unpack_profiles(manifest: dict[str, Any], section: str) -> dict[str, list[str]]:
    return {name: list(payload["features"]) for name, payload in manifest[section].items()}
