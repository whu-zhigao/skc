#!/usr/bin/env python3
"""Fail-fast verification for second-round ICDE headline numbers."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "systems"


def load(rel: str) -> dict:
    return json.loads((R / rel).read_text())


def close(actual, expected, tol, label):
    if abs(float(actual) - float(expected)) > tol:
        raise AssertionError(
            f"{label}: expected {expected}, got {actual}"
        )


def main() -> None:
    expected_f4 = {
        "diginetica": {
            "peak": 9.467308,
            "packed": 2.533900,
            "posting_peak": 6.931664,
            "recall": 0.680286,
            "dirty_hit": 0.695724,
            "checks": 88_647,
            "exact_hr": 0.164000,
            "posting_hr": 0.164333,
            "cached_hr": 0.164333,
            "tier2_mb": 12.400896,
            "tier2_checks": 48_441,
        },
        "gowalla": {
            "peak": 8.617352,
            "packed": 1.505728,
            "posting_peak": 7.110264,
            "recall": 0.579998,
            "dirty_hit": 0.472124,
            "checks": 77_292,
            "exact_hr": 0.093000,
            "posting_hr": 0.091333,
            "cached_hr": 0.091333,
            "tier2_mb": 5.481984,
            "tier2_checks": 21_414,
        },
    }
    for ds, exp in expected_f4.items():
        f4 = load(f"f4/{ds}_integrated_nq3000_p5_i8_theta0.1_s917.json")
        assert f4["n_queries"] == 3_000
        assert f4["packed"]["resident_mb"] <= 5.0 + 1e-6
        assert f4["posting_replay"]["resident_max_mb"] <= 8.0 + 1e-6
        close(f4["combined_peak_resident_mb"], exp["peak"], 0.02, f"{ds} peak")
        close(f4["packed"]["resident_mb"], exp["packed"], 0.02, f"{ds} packed")
        close(
            f4["posting_replay"]["resident_max_mb"], exp["posting_peak"],
            0.02, f"{ds} posting peak",
        )
        close(
            f4["posting_replay"]["recall_at_10"], exp["recall"], 2e-4,
            f"{ds} posting recall",
        )
        close(
            f4["drift"]["replay"]["packed_profile_hit_rate"],
            exp["dirty_hit"], 2e-4, f"{ds} dirty hit",
        )
        checks = (
            f4["initial"]["contract_checks"]
            + f4["drift"]["replay"]["contract_checks"]
            + f4["drift"]["refreshed_replay"]["contract_checks"]
        )
        violations = (
            f4["initial"]["contract_violations"]
            + f4["drift"]["replay"]["contract_violations"]
            + f4["drift"]["refreshed_replay"]["contract_violations"]
        )
        assert checks == exp["checks"]
        assert violations == 0
        quality = f4["quality"]
        close(
            quality["exact_neighbors_fresh"]["HR@20"], exp["exact_hr"],
            2e-4, f"{ds} exact HR",
        )
        close(
            quality["posting_neighbors_fresh"]["HR@20"], exp["posting_hr"],
            2e-4, f"{ds} posting HR",
        )
        close(
            quality["posting_neighbors_int8_cached"]["HR@20"],
            exp["cached_hr"], 2e-4, f"{ds} cached HR",
        )
        tier2 = f4["tier2_projection_audit"]
        close(tier2["fp32_projection_cache_mb"], exp["tier2_mb"], 0.02,
              f"{ds} Tier2 MB")
        assert tier2["checks"] == exp["tier2_checks"]
        assert tier2["violations"] == 0

    expected_dai = {
        "diginetica": (30, 36_780, 36_770, 36_770),
        "gowalla": (30, 69_420, 59_135, 53_291),
    }
    for ds, exp in expected_dai.items():
        data = load(f"dai2_pageledger_{ds}.json")
        theta = data["thetas"]["0.100"]
        local = theta["page_local"]
        assert data["n_intervals"] == exp[0]
        assert local["full_invalidation_refreshes"] == exp[1]
        assert local["total_page_refreshes"] == exp[2]
        assert theta["per_epoch"]["total_page_refreshes"] == exp[3]

    # Third-revision contract, publication, and source-placement checks.
    expected_contract = {
        "diginetica": (1_990_656, 2_016),
        "gowalla": (1_996_032, 1_944),
    }
    for ds, (checks, dirty) in expected_contract.items():
        contract = load(f"contract_matrix/{ds}.json")["summary"]
        assert contract["configurations"] == 54
        assert contract["profile_checks"] == checks
        assert contract["projection_checks"] == checks
        assert contract["profile_violations"] == 0
        assert contract["projection_violations"] == 0
        assert contract["false_clean"] == 0
        assert contract["dirty_decisions"] == dirty

    publication = load("publication_matrix.json")["summary"]
    assert publication == {
        "clean_reads": 625_891,
        "soft_misses": 168_237,
        "publications": 4_472,
        "contract_violations": 0,
    }

    gpu_publication = load("gpu_publication_matrix.json")
    assert gpu_publication["summary"] == {
        "clean_reads": 4_346,
        "soft_misses": 52_906,
        "torn_pages": 0,
        "version_mismatches": 0,
        "publications": 71_226,
    }
    assert gpu_publication["protocol"] == (
        "A800 CUDA default-stream optimizer updates; immutable source snapshot; "
        "reader CUDA streams; atomic CPU metadata pinning"
    )

    expected_global_item = {"diginetica": 36_780, "gowalla": 69_411}
    for ds, refreshes in expected_global_item.items():
        theta = load(f"dai2_pageledger_{ds}.json")["thetas"]["0.100"]
        assert theta["legacy_global_item_diagnostic"][
            "total_page_refreshes"
        ] == refreshes

    expected_source = {
        "diginetica": (12.626417, 0.691958, 10.279308),
        "gowalla": (9.076125, 0.694375, 7.157316),
    }
    for ds, (fresh, cache, resident_mb) in expected_source.items():
        route = load(f"source_placement/{ds}.json")["dimensions"]["512"]["128"]
        close(route["storage_fresh"]["p50_ms"], fresh, 0.01,
              f"{ds} storage source p50")
        close(route["int8_cache"]["p50_ms"], cache, 0.01,
              f"{ds} int8 cache p50")
        source_doc = load(f"source_placement/{ds}.json")
        close(
            source_doc["dimensions"]["512"]["offline_cache_resident_mb"],
            resident_mb, 0.01, f"{ds} offline cache MB",
        )

    expected_gpu_source = {
        "diginetica": (131.788908, 105.2048895, 24.8188995, 0.1316605, 0.99873048),
        "gowalla": (109.098216, 90.193095, 16.6401875, 0.123652, 0.99873054),
    }
    for ds, (storage, cache, mat, cached_mat, overlap) in expected_gpu_source.items():
        doc = load(f"gpu_source_serving/{ds}.json")
        b32 = doc["batch_results"]["32"]
        close(b32["storage"]["end_to_end_ms"]["p50_ms"], storage, 0.01,
              f"{ds} GPU storage p50")
        close(b32["int8_cache"]["end_to_end_ms"]["p50_ms"], cache, 0.01,
              f"{ds} GPU cache p50")
        close(b32["storage"]["materialize_ms"]["p50_ms"], mat, 0.01,
              f"{ds} GPU storage materialization")
        close(b32["int8_cache"]["materialize_ms"]["p50_ms"], cached_mat, 0.01,
              f"{ds} GPU cache materialization")
        close(doc["fidelity_batch32"]["int8_cache"]["top20_overlap_mean"],
              overlap, 1e-6, f"{ds} int8 top20 overlap")

    expected_full_model = {
        "diginetica": (30, 0.42835626121833514, 36_748),
        "gowalla": (30, 0.640745798971092, 67_805),
    }
    for ds, (changes, proj_l2, refreshes) in expected_full_model.items():
        full = load(f"full_model_drift/{ds}_seed1.json")
        assert full["summary"]["epochs_with_projection_view_change"] == changes
        close(full["summary"]["projection_weight_l2_mean"], proj_l2, 1e-6,
              f"{ds} full-model projection drift")
        replay = load(f"full_model_dai2/{ds}_seed1.json")["thetas"]["0.100"]
        assert replay["page_local"]["total_page_refreshes"] == refreshes

    f5d = load("f5/diginetica_power_heldout_g0.5_nqall_bs512.json")
    assert f5d["baseline"]["n_evaluated"] == 7_603
    assert f5d["baseline"]["hr20_hits"] == 1_171
    adv8 = min(
        (p for p in f5d["points"]
         if p["mode"] == "adversarial" and p["drift_l2"] == 0.8),
        key=lambda p: p["HR@20"],
    )
    close(adv8["hr_delta"], -0.00092, 2e-5, "Digi held-out adversarial")
    f5g = load("f5/gowalla_power_heldout_g0.5_nq20000_bs512.json")
    assert f5g["baseline"]["n_evaluated"] == 10_000
    assert f5g["baseline"]["hr20_hits"] == 1_072

    expected_f6 = {
        "diginetica": {
            "real": (22.12, 48.36, 281.95, 632.10, 212.06, 532.03),
            "concentrated": (20.00, 43.15, 39.23, 301.03, 198.80, 527.54),
        },
        "gowalla": {
            "real": (20.69, 44.69, 229.80, 635.39, 237.53, 592.86),
            "concentrated": (21.49, 47.30, 32.87, 265.86, 239.52, 600.64),
        },
    }
    for ds, scenarios in expected_f6.items():
        for scenario, exp in scenarios.items():
            f6 = load(
                f"f6/{ds}_sla_repeated_{scenario}_t12_q100_r2000_theta0.1.json"
            )
            for arm in ("none", "dai2", "full", "adaptive", "fresh"):
                assert f6["summary"][arm]["n_traces"] == 12
            close(f6["summary"]["fresh"]["p50_ms"]["mean"], exp[0], 0.02,
                  f"{ds}/{scenario} fresh p50")
            close(f6["summary"]["fresh"]["p99_ms"]["mean"], exp[1], 0.02,
                  f"{ds}/{scenario} fresh p99")
            close(f6["summary"]["dai2"]["p50_ms"]["mean"], exp[2], 0.02,
                  f"{ds}/{scenario} DAI p50")
            close(f6["summary"]["dai2"]["p99_ms"]["mean"], exp[3], 0.02,
                  f"{ds}/{scenario} DAI p99")
            close(f6["summary"]["full"]["p50_ms"]["mean"], exp[4], 0.02,
                  f"{ds}/{scenario} full p50")
            close(f6["summary"]["full"]["p99_ms"]["mean"], exp[5], 0.02,
                  f"{ds}/{scenario} full p99")

    posting = {
        "diginetica": {
            "real": (8_791.42, 16_241.65, 18_210.58, 33_861.06),
            "concentrated": (9_040.87, 16_762.16, 13_374.34, 24_389.88),
        },
        "gowalla": {
            "real": (16_658.33, 28_834.13, 30_789.74, 55_104.39),
            "concentrated": (
                16_915.21, 29_427.53, 23_748.98, 40_639.77
            ),
        },
    }
    for ds, scenarios in posting.items():
        for scenario, exp in scenarios.items():
            f6 = load(
                f"f6/{ds}_sla_repeated_{scenario}_posting_"
                f"t10_q100_r500_theta0.1.json"
            )
            close(f6["summary"]["fresh"]["p50_ms"]["mean"], exp[0], 10.0,
                  f"{ds}/{scenario}/posting fresh p50")
            close(f6["summary"]["fresh"]["p99_ms"]["mean"], exp[1], 10.0,
                  f"{ds}/{scenario}/posting fresh p99")
            close(f6["summary"]["dai2"]["p50_ms"]["mean"], exp[2], 10.0,
                  f"{ds}/{scenario}/posting DAI p50")
            close(f6["summary"]["dai2"]["p99_ms"]["mean"], exp[3], 10.0,
                  f"{ds}/{scenario}/posting DAI p99")

    print("second-round paper headline numbers: verified")
    print("third-revision contract/source numbers: verified")


if __name__ == "__main__":
    main()
