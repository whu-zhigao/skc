#!/usr/bin/env python3
"""Run the high-priority ICDE revision experiment suite.

Profiles:
  smoke : CPU/local sanity checks in minutes.
  core  : prior A800 evidence suite.
  revision2 : address the second-round review with actual-model drift,
              fresh-on-demand, adaptive DAI-2, posting-in-path SLA, Tier-2
              projection checks, and end-to-end ranking quality.
  full  : core plus posting-in-path SLA and complete-test calibration.

Examples
--------
    python3 experiments_icde/run_revision_experiments.py --profile smoke
    python3 experiments_icde/run_revision_experiments.py --profile core
    python3 experiments_icde/run_revision_experiments.py --profile full
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def cmd(parts):
    return [PY, str(ROOT / "experiments_icde" / parts[0])] + parts[1:]


def build_jobs(profile: str, datasets: list[str]) -> list[tuple[str, list[str]]]:
    jobs: list[tuple[str, list[str]]] = []
    tests = [
        ("tests", [PY, str(ROOT / "tests" / "test_hot_page_contract.py")]),
        ("tests", [PY, str(ROOT / "tests" / "test_bounded_residency.py")]),
        ("tests", [PY, str(ROOT / "tests" / "test_page_consistency.py")]),
    ]
    jobs.extend(tests)

    if profile == "smoke":
        for ds in datasets:
            data = str(ROOT / "data" / f"{ds}.pkl")
            jobs.append((
                f"f4-smoke-{ds}",
                ["exp_f4_integrated.py", "--data", data,
                 "--n-query", "100", "--drift-items", "8", "--force"],
            ))
        jobs.append((
            "f5-smoke",
            ["exp_f5_contract_power.py",
             "--data", str(ROOT / "data" / "diginetica.pkl"),
             "--n-query", "256", "--batch-size", "128",
             "--drifts", "0.1", "--fractions", "1.0",
             "--seeds", "0", "--modes", "random,correlated,adversarial",
             "--bootstrap", "100", "--force"],
        ))
        jobs.append((
            "f6-smoke",
            ["exp_f6_sla_repeated.py",
             "--data", str(ROOT / "data" / "diginetica.pkl"),
             "--n-query", "64", "--requests", "32", "--traces", "1",
             "--batch-size", "32", "--maintenance-interval-ms", "5000",
             "--dirty-window-ms", "0", "--drift-items", "2",
             "--bootstrap", "50", "--force"],
        ))
        jobs.append((
            "a3-smoke",
            ["exp_a3_model_drift.py",
             "--data", str(ROOT / "data" / "diginetica.pkl"),
             "--epochs", "2", "--steps-per-epoch", "2",
             "--batch-size", "8", "--force"],
        ))
        return jobs

    if profile == "revision2":
        for ds in datasets:
            data = str(ROOT / "data" / f"{ds}.pkl")
            saved = str(
                ROOT / "results" / "real" / f"{ds}_a3_cotrain_seed0.pt"
            )
            jobs.append((
                f"a3-actual-model-{ds}",
                ["exp_a3_model_drift.py", "--data", data,
                 "--epochs", "30", "--steps-per-epoch", "40",
                 "--batch-size", "256", "--seed", "0",
                 "--save-model", saved, "--force"],
            ))

        for ds in datasets:
            data = str(ROOT / "data" / f"{ds}.pkl")
            trace = str(
                ROOT / "results" / "systems" / "a3" /
                f"{ds}_mpli_e30_s40_seed0.npz"
            )
            jobs.append((
                f"dai2-actual-model-{ds}",
                [str(ROOT / "systems" / "bench_dai2_pageledger.py"),
                 "--data", data, "--drift", trace,
                 "--thetas", "0.05,0.10,0.15"],
            ))

        for ds in datasets:
            data = str(ROOT / "data" / f"{ds}.pkl")
            saved = str(
                ROOT / "results" / "real" / f"{ds}_a3_cotrain_seed0.pt"
            )
            jobs.append((
                f"f4-integrated-quality-{ds}",
                ["exp_f4_integrated.py", "--data", data,
                 "--ckpt", saved, "--n-query", "3000",
                 "--quality-batch-size", "256", "--drift-items", "80",
                 "--evict-after-read", "--force"],
            ))

        f5_nq = {"diginetica": "0", "gowalla": "20000"}
        for ds in datasets:
            saved = str(
                ROOT / "results" / "real" / f"{ds}_a3_cotrain_seed0.pt"
            )
            jobs.append((
                f"f5-heldout-operating-{ds}",
                ["exp_f5_contract_power.py",
                 "--data", str(ROOT / "data" / f"{ds}.pkl"),
                 "--ckpt", saved, "--n-query", f5_nq.get(ds, "5000"),
                 "--batch-size", "512", "--gradient-fraction", "0.5",
                 "--modes", "random,correlated,adversarial,margin",
                 "--force"],
            ))

        for scenario in ("real", "concentrated"):
            for ds in datasets:
                data = str(ROOT / "data" / f"{ds}.pkl")
                trace = str(
                    ROOT / "results" / "systems" / "a3" /
                    f"{ds}_mpli_e30_s40_seed0.npz"
                )
                jobs.append((
                    f"f6-{scenario}-fresh-adaptive-{ds}",
                    ["exp_f6_sla_repeated.py", "--data", data,
                     "--ckpt", str(
                         ROOT / "results" / "real" /
                         f"{ds}_a3_cotrain_seed0.pt"
                     ),
                     "--n-query", "1000", "--requests", "2000",
                     "--traces", "12", "--target-qps", "100",
                     "--batch-size", "32", "--modes",
                     "none,dai2,full,adaptive,fresh",
                     "--maintenance-interval-ms", "500",
                     "--dirty-window-ms", "2", "--drift-scenario", scenario,
                     "--drift-trace", trace, "--force"],
                ))

        for scenario in ("real", "concentrated"):
            for ds in datasets:
                data = str(ROOT / "data" / f"{ds}.pkl")
                trace = str(
                    ROOT / "results" / "systems" / "a3" /
                    f"{ds}_mpli_e30_s40_seed0.npz"
                )
                jobs.append((
                    f"f6-{scenario}-posting-{ds}",
                    ["exp_f6_sla_repeated.py", "--data", data,
                     "--ckpt", str(
                         ROOT / "results" / "real" /
                         f"{ds}_a3_cotrain_seed0.pt"
                     ),
                     "--n-query", "500", "--requests", "500",
                     "--traces", "10", "--target-qps", "100",
                     "--batch-size", "32", "--modes", "fresh,dai2,adaptive",
                     "--use-posting", "--maintenance-interval-ms", "500",
                     "--dirty-window-ms", "2", "--drift-scenario", scenario,
                     "--drift-trace", trace, "--force"],
                ))
        return jobs

    # Longer real drift traces. Sixty intervals avoid headline conclusions
    # based on Diginetica's previous three-interval trace.
    for ds in datasets:
        jobs.append((
            f"a2-long-{ds}",
            ["exp_a2_drift_trace.py", "--data", str(ROOT / "data" / f"{ds}.pkl"),
             "--steps", "3000", "--snap-every", "50", "--force"],
        ))

    for ds in datasets:
        data = str(ROOT / "data" / f"{ds}.pkl")
        trace = str(ROOT / "results" / "systems" / f"drift_trace_{ds}.npz")
        jobs.append((
            f"dai2-long-{ds}",
            [str(ROOT / "systems" / "bench_dai2_pageledger.py"),
             "--data", data, "--drift", trace,
             "--thetas", "0.05,0.10,0.15"],
        ))

    for ds in datasets:
        data = str(ROOT / "data" / f"{ds}.pkl")
        jobs.append((
            f"f4-integrated-{ds}",
            ["exp_f4_integrated.py", "--data", data,
             "--n-query", "3000", "--drift-items", "80",
             "--evict-after-read", "--force"],
        ))

    f5_nq = {"diginetica": "0", "gowalla": "10000"}
    for ds in datasets:
        jobs.append((
            f"f5-power-{ds}",
            ["exp_f5_contract_power.py",
             "--data", str(ROOT / "data" / f"{ds}.pkl"),
             "--n-query", f5_nq.get(ds, "5000"),
             "--batch-size", "512", "--force"],
        ))

    for ds in datasets:
        data = str(ROOT / "data" / f"{ds}.pkl")
        trace = str(ROOT / "results" / "systems" / f"drift_trace_{ds}.npz")
        jobs.append((
            f"f6-sla-{ds}",
            ["exp_f6_sla_repeated.py", "--data", data,
             "--n-query", "1000", "--requests", "2000",
             "--traces", "30", "--target-qps", "100",
             "--batch-size", "32", "--maintenance-interval-ms", "500",
             "--dirty-window-ms", "2", "--drift-trace", trace,
             "--force"],
        ))

    if profile == "full":
        for ds in datasets:
            data = str(ROOT / "data" / f"{ds}.pkl")
            trace = str(ROOT / "results" / "systems" / f"drift_trace_{ds}.npz")
            jobs.append((
                f"f6-sla-posting-{ds}",
                ["exp_f6_sla_repeated.py", "--data", data,
                 "--n-query", "1000", "--requests", "1000",
                 "--traces", "10", "--target-qps", "100",
                 "--batch-size", "32", "--use-posting",
                 "--maintenance-interval-ms", "500",
                 "--dirty-window-ms", "2", "--drift-trace", trace,
                 "--force"],
            ))
            jobs.append((
                f"f5-full-{ds}",
                ["exp_f5_contract_power.py",
                 "--data", data, "--n-query", "0",
                 "--batch-size", "512", "--force"],
            ))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile",
                    choices=["smoke", "core", "revision2", "full"],
                    default="core")
    ap.add_argument("--datasets", default="diginetica,gowalla")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    args = ap.parse_args()

    if args.profile in {"core", "revision2", "full"} and not args.dry_run:
        try:
            subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                            "--format=csv,noheader"], check=True)
        except Exception as exc:
            raise SystemExit(
                f"core/full profiles require CUDA; nvidia-smi failed: {exc}"
            )

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    jobs = build_jobs(args.profile, datasets)
    for _name, command in jobs:
        if command and command[0].endswith(".py"):
            script = Path(command[0])
            resolved = script if script.is_absolute() else (
                ROOT / "experiments_icde" / script
            )
            command[:] = [PY, str(resolved)] + command[1:]
        elif command and command[0] == PY and not Path(command[1]).is_absolute():
            command[1] = str(ROOT / "experiments_icde" / command[1])
    if args.dry_run:
        for name, command in jobs:
            print(f"{name}: {' '.join(command)}")
        print(f"total jobs: {len(jobs)}")
        return

    failures = []
    started = time.perf_counter()
    for i, (name, command) in enumerate(jobs, 1):
        print(f"\n=== [{i}/{len(jobs)}] {name} ===", flush=True)
        print(" ".join(command), flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run(command, cwd=ROOT)
        elapsed = time.perf_counter() - t0
        status = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
        print(f"=== {name}: {status} in {elapsed:.1f}s ===", flush=True)
        if proc.returncode:
            failures.append((name, proc.returncode))
            if not args.continue_on_error:
                break

    print(f"\nrevision suite finished in {time.perf_counter() - started:.1f}s")
    if failures:
        print("failures:")
        for name, code in failures:
            print(f"  {name}: exit {code}")
        raise SystemExit(1)
    print("all revision jobs passed")


if __name__ == "__main__":
    main()
