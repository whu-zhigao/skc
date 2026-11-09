#!/usr/bin/env python3
"""Generate paper macros from the second-round A800 revision results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "systems"
PAPER = ROOT.parent / "paper"


def load(rel: str) -> dict:
    return json.loads((R / rel).read_text())


def fmt(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def int_fmt(value: int) -> str:
    return f"{int(value):,}".replace(",", "{,}")


def add_f4(prefix: str, dataset: str, macros: Dict[str, str]) -> None:
    data = load(
        f"f4/{dataset}_integrated_nq3000_p5_i8_theta0.1_s917.json"
    )
    peak = data["combined_peak_resident_mb"]
    macros[f"Ffour{prefix}PeakResidentMB"] = fmt(peak)
    macros[f"Ffour{prefix}ResidentMB"] = fmt(peak)
    macros[f"Ffour{prefix}PackedResidentMB"] = fmt(
        data["packed"]["resident_mb"]
    )
    macros[f"Ffour{prefix}PostingPeakResidentMB"] = fmt(
        data["posting_replay"]["resident_max_mb"]
    )
    macros[f"Ffour{prefix}PostingEndResidentMB"] = fmt(
        data["posting"]["resident_end_mb"]
    )
    macros[f"Ffour{prefix}PostingRecall"] = fmt(
        data["posting_replay"]["recall_at_10"], 3
    )
    macros[f"Ffour{prefix}CleanHit"] = fmt(
        data["initial"]["packed_profile_hit_rate"], 3
    )
    macros[f"Ffour{prefix}ProfileHit"] = macros[f"Ffour{prefix}CleanHit"]
    macros[f"Ffour{prefix}DirtyHit"] = fmt(
        data["drift"]["replay"]["packed_profile_hit_rate"], 3
    )
    checks = (
        data["initial"]["contract_checks"]
        + data["drift"]["replay"]["contract_checks"]
        + data["drift"]["refreshed_replay"]["contract_checks"]
    )
    violations = (
        data["initial"]["contract_violations"]
        + data["drift"]["replay"]["contract_violations"]
        + data["drift"]["refreshed_replay"]["contract_violations"]
    )
    macros[f"Ffour{prefix}Checks"] = int_fmt(checks)
    macros[f"Ffour{prefix}ContractChecks"] = macros[f"Ffour{prefix}Checks"]
    macros[f"Ffour{prefix}Violations"] = str(violations)
    macros[f"Ffour{prefix}ContractViolations"] = str(violations)
    macros[f"Ffour{prefix}PostingPFifty"] = fmt(
        data["posting_replay"]["p50_ms"], 1
    )
    macros[f"Ffour{prefix}PostingPNinetyNine"] = fmt(
        data["posting_replay"]["p99_ms"], 1
    )
    macros[f"Ffour{prefix}ActualDisp"] = fmt(
        data["initial"]["actual_profile_dispersion_max"], 3
    )
    macros[f"Ffour{prefix}ReportedBound"] = fmt(
        data["initial"]["reported_bound_max"], 3
    )
    macros[f"Ffour{prefix}ColdMB"] = fmt(data["posting"]["cold_storage_mb"], 1)
    macros[f"Ffour{prefix}RSSMB"] = fmt(data["process_maxrss_mb"], 1)

    quality = data["quality"]
    macros[f"Ffour{prefix}ExactHR"] = fmt(
        100 * quality["exact_neighbors_fresh"]["HR@20"]
    )
    macros[f"Ffour{prefix}PostingFreshHR"] = fmt(
        100 * quality["posting_neighbors_fresh"]["HR@20"]
    )
    macros[f"Ffour{prefix}PostingCachedHR"] = fmt(
        100 * quality["posting_neighbors_int8_cached"]["HR@20"]
    )

    tier2 = data["tier2_projection_audit"]
    macros[f"Ffour{prefix}TierTwoMB"] = fmt(
        tier2["fp32_projection_cache_mb"]
    )
    macros[f"Ffour{prefix}TierTwoActual"] = fmt(
        tier2["actual_max_projection_drift"], 3
    )
    macros[f"Ffour{prefix}TierTwoBound"] = fmt(
        tier2["operator_norm_bound_max"], 3
    )
    macros[f"Ffour{prefix}TierTwoChecks"] = int_fmt(tier2["checks"])
    macros[f"Ffour{prefix}TierTwoViolations"] = str(tier2["violations"])
    macros[f"Ffour{prefix}WithTierTwoMB"] = fmt(
        peak + tier2["fp32_projection_cache_mb"]
    )


def add_dai(prefix: str, dataset: str, macros: Dict[str, str]) -> None:
    data = load(f"dai2_pageledger_{dataset}.json")
    theta = data["thetas"]["0.100"]
    local = theta["page_local"]
    macros[f"Dai{prefix}Intervals"] = str(data["n_intervals"])
    macros[f"Dai{prefix}Pages"] = str(local["pages"])
    macros[f"Dai{prefix}FullRefreshes"] = int_fmt(
        local["full_invalidation_refreshes"]
    )
    macros[f"Dai{prefix}LocalRefreshes"] = int_fmt(
        local["total_page_refreshes"]
    )
    macros[f"Dai{prefix}PerEpochRefreshes"] = int_fmt(
        theta["per_epoch"]["total_page_refreshes"]
    )
    macros[f"Dai{prefix}SavedPercent"] = fmt(
        100 * (1 - local["total_page_refreshes"]
               / local["full_invalidation_refreshes"])
    )
    macros[f"Dai{prefix}MembershipMB"] = fmt(local["membership_mb"])
    macros[f"Dai{prefix}LedgerMB"] = fmt(local["ledger_mb"])


def add_f5(prefix: str, dataset: str, macros: Dict[str, str]) -> None:
    candidates = [
        f"f5/{dataset}_power_heldout_g0.5_nqall_bs512.json",
        f"f5/{dataset}_power_heldout_g0.5_nq20000_bs512.json",
    ]
    filename = next(name for name in candidates if (R / name).exists())
    data = load(filename)
    macros[f"Ffive{prefix}GradientQueries"] = int_fmt(
        data["gradient_split_queries"]
    )
    macros[f"Ffive{prefix}HeldoutQueries"] = int_fmt(
        data["heldout_eval_queries"]
    )
    macros[f"Ffive{prefix}BaseHits"] = int_fmt(data["baseline"]["hr20_hits"])
    macros[f"Ffive{prefix}BaseHR"] = fmt(
        100 * data["baseline"]["HR@20"]
    )
    for mode, label in (
        ("random", "Random"), ("correlated", "Correlated"),
        ("adversarial", "Adversarial"), ("margin", "Margin"),
    ):
        points = [p for p in data["points"] if p["mode"] == mode]
        point = max(points, key=lambda p: p["drift_l2"])
        macros[f"Ffive{prefix}{label}Delta"] = fmt(
            100 * point["hr_delta"]
        )
        macros[f"Ffive{prefix}{label}CI"] = (
            f"[{fmt(100*point['hr_delta_ci95'][0])}, "
            f"{fmt(100*point['hr_delta_ci95'][1])}]"
        )
    adv_one = next(
        p for p in data["points"]
        if p["mode"] == "adversarial" and p["drift_l2"] == 0.1
    )
    macros[f"Ffive{prefix}AdversarialDeltaPointOne"] = fmt(
        100 * adv_one["hr_delta"]
    )


def add_f6(prefix: str, dataset: str, macros: Dict[str, str]) -> None:
    files = {
        "Real": f"{dataset}_sla_repeated_real_t12_q100_r2000_theta0.1.json",
        "Concentrated": (
            f"{dataset}_sla_repeated_concentrated_t12_q100_r2000_theta0.1.json"
        ),
        "RealPosting": (
            f"{dataset}_sla_repeated_real_posting_t10_q100_r500_theta0.1.json"
        ),
        "ConcentratedPosting": (
            f"{dataset}_sla_repeated_concentrated_posting_"
            f"t10_q100_r500_theta0.1.json"
        ),
    }
    arm_labels = {
        "none": "NoMaint", "dai2": "Dai", "full": "Full",
        "adaptive": "Adaptive", "fresh": "Fresh",
    }
    for scenario, filename in files.items():
        data = load(f"f6/{filename}")
        macros[f"Fsix{scenario}{prefix}Traces"] = str(
            data["summary"]["dai2"]["n_traces"]
        )
        for arm, label in arm_labels.items():
            if arm not in data["summary"]:
                continue
            summary = data["summary"][arm]
            stem = f"Fsix{scenario}{prefix}{label}"
            macros[f"{stem}PFifty"] = fmt(summary["p50_ms"]["mean"], 1)
            macros[f"{stem}PNinetyNine"] = fmt(
                summary["p99_ms"]["mean"], 1
            )
            macros[f"{stem}PNinetyNineCI"] = (
                f"[{fmt(summary['p99_bootstrap_ci95'][0], 1)}, "
                f"{fmt(summary['p99_bootstrap_ci95'][1], 1)}]"
            )
            macros[f"{stem}QPS"] = fmt(
                summary["achieved_qps"]["mean"], 1
            )
            macros[f"{stem}Refresh"] = fmt(
                summary["refresh_ms_mean"]["mean"], 1
            )
    macros[f"Fsix{prefix}Traces"] = macros["FsixReal" + prefix + "Traces"]
    macros[f"Fsix{prefix}DaiPNinetyNine"] = macros[
        f"FsixReal{prefix}DaiPNinetyNine"
    ]
    macros[f"Fsix{prefix}FullPNinetyNine"] = macros[
        f"FsixReal{prefix}FullPNinetyNine"
    ]
    macros[f"Fsix{prefix}FreshPNinetyNine"] = macros[
        f"FsixReal{prefix}FreshPNinetyNine"
    ]


def latex_macros() -> Dict[str, str]:
    macros: Dict[str, str] = {}
    for dataset, prefix in (
        ("diginetica", "Digi"), ("gowalla", "Gowa"),
    ):
        add_f4(prefix, dataset, macros)
        add_dai(prefix, dataset, macros)
        add_f5(prefix, dataset, macros)
        add_f6(prefix, dataset, macros)
    return macros


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=PAPER / "revision_generated.tex")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    macros = latex_macros()
    if args.check_only:
        for key, value in macros.items():
            print(f"{key}={value}")
        return
    lines = [
        "% Auto-generated by analysis/summarize_revision_results.py.",
        "% Do not edit by hand; regenerate after A3/F4/F5/F6 complete.",
    ]
    lines.extend(
        f"\\newcommand{{\\{key}}}{{{value}}}"
        for key, value in macros.items()
    )
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} with {len(macros)} macros")


if __name__ == "__main__":
    main()
