#!/usr/bin/env python3
"""Parallel driver for the ICDE experiment suite (stdlib only).

Builds the job matrix (stage x dataset x seed x the stage's default key
flags), runs it on two queues -- a CPU worker pool and a small GPU slot
pool -- logs each job to logs/<tag>.log, skips jobs whose inputs are
missing (with a log line instead of dying), and ends with a manifest
JSON listing every job with status / wall time / output files.

Usage examples:
    python3 run_parallel.py --dry-run              # print the matrix
    python3 run_parallel.py                        # full paper matrix
    python3 run_parallel.py --smoke                # tiny caps, minutes
    python3 run_parallel.py --jobs a1,b1 --datasets gowalla --seeds 0
    python3 run_parallel.py --jobs b6 --force      # rerun one stage

Resume: every experiment writes one JSON per configuration under
results/systems/<exp>/<tag>.json and skips configurations that already
exist (pass --force to rerun); SIGINT drains in-flight jobs gracefully
and still writes the manifest."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

SUITE = Path(__file__).resolve().parent
sys.path.insert(0, str(SUITE))
from common import CODE, RESULTS, data_file, ensure_results  # noqa: E402

REAL_DIR = CODE / "results" / "real"

# ---------------------------------------------------------------------------
# CLASSIFICATION -- edit these dicts to move stages between the queues.
# GPU_STAGES: experiments that train or run model forwards and should
# occupy the scarce --gpu-workers slots (A800). a2 counts as GPU only in
# its inline-training mode (without --ckpts); with --ckpts it is CPU
# trace extraction. Everything else runs in the --cpu-workers pool.
GPU_STAGES = {"b2", "b7", "f5", "f6"}
# Stages that need a checkpoint (default: results/real/abl_no_residual_
# <dataset>.pt, overridable per job via --ckpt / CKPT-like flags).
NEEDS_CKPT = {"b2", "b3", "b4", "b6", "b7", "c2", "d1", "d2", "d3",
              "f4", "f5", "f6"}
# Stages that need an OLD_CKPT env var (earlier-generation checkpoint).
NEEDS_OLD_CKPT = {"b4", "b7"}
# Stages that need the A2 drift trace npz for the same dataset.
NEEDS_TRACE = {"b3", "c1"}
# Same-dataset ordering: value stage must finish before key stage starts.
DEPENDS = {"b3": "a2", "c1": "a2"}
# Stages accepting --seed (others are deterministic; the seed dimension
# collapses to a single job for them).
SEEDED = {"a2", "b5", "b6", "b7", "d4"}
# Stages pinned to their own dataset regardless of --datasets (d2 is
# the Yoochoose-1/4 scaling study; CPU class, no GPU slots).
DATASETS_BY_STAGE = {"d2": ["yoochoose_quarter"]}

STAGES = ["a1", "a2", "b1", "b2", "b3", "b4", "b5", "b6", "b7",
          "c1", "c2", "d1", "d3", "d4", "d2", "f4", "f5", "f6"]
DATASETS = ["diginetica", "gowalla"]
SEEDS = [0, 1, 2]
# Default key flags per stage for the full matrix (edit freely).
STAGE_FLAGS = {
    "a1": [], "a2": [], "b1": [], "b2": [], "b3": ["--live"],
    "b4": [], "b5": [], "b6": [], "b7": ["--train"], "c1": [], "c2": [],
    "d1": [], "d3": [], "d4": [], "d2": [],
    "f4": ["--n-query", "3000", "--evict-after-read"],
    "f5": ["--batch-size", "512"],
    "f6": ["--requests", "2000", "--traces", "30"],
}
# --smoke profile: 1 dataset, 1 seed, tiny caps (existing flags only).
SMOKE_FLAGS = {
    "a1": ["--n-query", "50"],
    "a2": ["--steps", "60", "--snap-every", "30"],
    "b1": ["--n-query", "50"],
    "b2": ["--n-query", "20", "--batch-size", "64"],
    "b3": [],
    "b4": ["--n-query", "20"],
    "b5": ["--n-query", "40", "--greedy-subsample", "10",
           "--total-mb", "5", "--splits", "uniform"],
    "b6": ["--n-query", "20", "--configs", "int8-all,utility-mixed"],
    "b7": ["--ratios", "0,100", "--n-query", "10", "--train",
           "--epochs", "1", "--steps-per-epoch", "3",
           "--refresh-every", "1,inf", "--cache-mb", "0.05"],
    "c1": [],
    "c2": [],
    "d1": ["--n-query", "30"],
    "d3": ["--seconds", "1.0"],
    "d4": ["--n-inserts", "500", "--n-queries", "100"],
    "d2": ["--n-query", "50", "--ann-corpus-cap", "5000"],
    "f4": ["--n-query", "100", "--drift-items", "8"],
    "f5": ["--n-query", "256", "--batch-size", "128",
           "--drifts", "0.1", "--fractions", "1.0", "--seeds", "0",
           "--modes", "random,correlated,adversarial", "--bootstrap", "100"],
    "f6": ["--n-query", "64", "--requests", "32", "--traces", "1",
           "--batch-size", "32", "--maintenance-interval-ms", "5000",
           "--dirty-window-ms", "0", "--drift-items", "2",
           "--bootstrap", "50"],
}


def build_jobs(stages, datasets, seeds, smoke, force, extra_flags):
    jobs = []
    for stage in stages:
        s_seeds = seeds if stage in SEEDED else [None]
        sflags = list(SMOKE_FLAGS.get(stage, [])) if smoke \
            else list(STAGE_FLAGS.get(stage, []))
        if stage in extra_flags:
            sflags += extra_flags[stage]
        for ds in DATASETS_BY_STAGE.get(stage, datasets):
            for seed in s_seeds:
                tag = f"{stage}_{ds}" + (f"_s{seed}"
                                         if seed is not None else "")
                if smoke:
                    tag += "_smoke"
                flags = list(sflags)
                if seed is not None:
                    flags += ["--seed", str(seed)]
                if force:
                    flags.append("--force")
                jobs.append({
                    "stage": stage, "dataset": ds, "seed": seed,
                    "tag": tag, "flags": flags,
                    "kind": "gpu" if _is_gpu(stage, flags) else "cpu"})
    return jobs


def _is_gpu(stage, flags):
    if stage in GPU_STAGES:
        return True
    if stage == "a2" and "--ckpts" not in flags:
        return True        # inline training mode uses CUDA when present
    return False


def job_cmd(job):
    script = SUITE / f"exp_{job['stage']}_*.py"
    matches = sorted(SUITE.glob(script.name))
    if not matches:
        return None
    cmd = [sys.executable, str(matches[0]),
           "--data", str(data_file(job["dataset"]))]
    cmd += job["flags"]
    if job["stage"] == "b4":
        cmd += ["--ckpt", _default_ckpt(job["dataset"]),
                "--old-ckpt", _old_ckpt(job["dataset"])]
    elif job["stage"] == "b7":
        cmd += ["--new-ckpt", _default_ckpt(job["dataset"]),
                "--old-ckpt", _old_ckpt(job["dataset"])]
    elif job["stage"] in NEEDS_CKPT and "--ckpt" not in job["flags"]:
        # b2/b3/b6/c2/d1/d2/d3 accept --ckpt; without it they fall back
        # to the DIGINETICA default in common.load_model and crash on
        # any other dataset's vocab (gowalla 30001 vs 38250).
        cmd += ["--ckpt", _default_ckpt(job["dataset"])]
    return cmd


def _default_ckpt(ds):
    """NEW-generation ckpt: abl_no_residual_<ds>.pt if present, else the
    full mplignn_<ds>.pt (same 47-key architecture, per-dataset item
    count; the experiment JSON records the path actually used)."""
    for name in (f"abl_no_residual_{ds}.pt", f"mplignn_{ds}.pt"):
        p = REAL_DIR / name
        if p.exists():
            return str(p)
    return str(REAL_DIR / f"abl_no_residual_{ds}.pt")


def _old_ckpt(ds):
    """OLD-generation ckpt for b4/b7: env OLD_CKPT first (a '{ds}'
    template is expanded per dataset), else the per-dataset default
    pair if present on disk."""
    env = os.environ.get("OLD_CKPT", "")
    if env:
        return env.replace("{ds}", ds)
    for name in (f"abl_no_anchor_{ds}.pt", f"abl_nodenoise_{ds}.pt"):
        p = REAL_DIR / name
        if p.exists():
            return str(p)
    return ""


def missing_deps(job):
    """Inputs that must exist before the job can run (else skip)."""
    miss = []
    pkl = data_file(job["dataset"])
    if not pkl.exists():
        miss.append(f"dataset {pkl}")
    if job["stage"] in NEEDS_CKPT and "--ckpt" not in job["flags"]:
        ck = Path(_default_ckpt(job["dataset"]))
        if not ck.exists():
            miss.append(f"ckpt {ck} (pass a --ckpt flag or place the "
                        f"default)")
    if job["stage"] in NEEDS_OLD_CKPT:
        old = _old_ckpt(job["dataset"])
        if not old:
            miss.append("old ckpt (export OLD_CKPT=... or place "
                        f"results/real/abl_no_anchor_{job['dataset']}.pt"
                        " / abl_nodenoise_<ds>.pt)")
        elif not Path(old).exists():
            miss.append(f"OLD_CKPT {old}")
    if job["stage"] in NEEDS_TRACE:
        npz = ensure_results() / f"drift_trace_{job['dataset']}.npz"
        if not npz.exists():
            miss.append(f"trace {npz} (run stage a2 first)")
    return miss


def snapshot_results():
    snap = {}
    if RESULTS.exists():
        for p in RESULTS.rglob("*"):
            if p.is_file():
                snap[str(p)] = p.stat().st_mtime
    return snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="all",
                    help="comma list of stages, or 'all'")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--cpu-workers", type=int,
                    default=min(12, max(1, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--gpu-workers", type=int, default=2)
    ap.add_argument("--force", action="store_true",
                    help="pass --force to every experiment (rerun "
                         "existing per-config JSONs)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny caps on 1 dataset / 1 seed (minutes)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the job matrix and exit")
    ap.add_argument("--manifest", default=None,
                    help="manifest path (default results/systems/"
                         "parallel_manifest_<ts>.json)")
    args = ap.parse_args()

    stages = (STAGES if args.jobs == "all"
              else [s.strip() for s in args.jobs.split(",")])
    bad = [s for s in stages if s not in STAGES]
    if bad:
        sys.exit(f"[run_parallel] unknown stages: {bad} "
                 f"(valid: {STAGES})")
    datasets = [d.strip() for d in args.datasets.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    extra_flags = {}   # reserved: per-stage extra flags via STAGE_FLAGS

    if args.smoke:
        datasets, seeds = [datasets[0]], [seeds[0]]
        args.cpu_workers = min(args.cpu_workers, 4)
        args.gpu_workers = 1

    jobs = build_jobs(stages, datasets, seeds, args.smoke, args.force,
                      extra_flags)
    if args.dry_run:
        for j in jobs:
            deps = missing_deps(j)
            print(f"{j['kind']:>3s} {j['tag']:<28s} "
                  f"{'MISSING: ' + '; '.join(deps) if deps else 'ok'}")
        print(f"{len(jobs)} jobs "
              f"(cpu pool={args.cpu_workers}, gpu slots={args.gpu_workers})")
        return

    logs = CODE / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    ensure_results()
    env = os.environ.copy()
    env.setdefault("SKC_CODE_ROOT", str(CODE))   # passthrough + default
    # CUDA_VISIBLE_DEVICES and friends pass through via os.environ.copy()

    stop = {"flag": False}

    def on_sigint(sig, frame):
        if stop["flag"]:
            signal.default_int_handler(sig, frame)   # second Ctrl-C aborts
        stop["flag"] = True
        print("\n[run_parallel] SIGINT: finishing in-flight jobs, not "
              "submitting new ones (Ctrl-C again to abort)...",
              flush=True)

    signal.signal(signal.SIGINT, on_sigint)

    pools = {"cpu": ThreadPoolExecutor(max_workers=args.cpu_workers),
             "gpu": ThreadPoolExecutor(max_workers=args.gpu_workers)}
    records = {j["tag"]: dict(j, status="pending", rc=None, wall_s=None,
                              log=None, outputs=[]) for j in jobs}
    futures = {}

    def deps_done(job):
        dep = DEPENDS.get(job["stage"])
        if dep is None:
            return True
        dep_tags = [t for t, r in records.items()
                    if r["stage"] == dep and r["dataset"] == job["dataset"]]
        if not dep_tags:
            return True    # dep stage not in this matrix; the trace-npz
                           # existence check in missing_deps covers it
        return all(records[t]["status"] in ("done", "skipped_dep",
                                            "failed")
                   for t in dep_tags)

    def run_job(job):
        rec = records[job["tag"]]
        before = snapshot_results()
        log = logs / f"{job['tag']}.log"
        rec["log"] = str(log)
        t0 = time.perf_counter()
        with open(log, "w") as fh:
            fh.write(f"# {job['tag']} ({job['kind']})\n"
                     f"# cmd: {' '.join(job_cmd(job) or [])}\n")
            fh.flush()
            try:
                r = subprocess.run(job_cmd(job), cwd=str(CODE), env=env,
                                   stdout=fh, stderr=subprocess.STDOUT)
                rec["rc"] = r.returncode
                rec["status"] = "done" if r.returncode == 0 else "failed"
            except Exception as e:                     # noqa: BLE001
                fh.write(f"\n[run_parallel] spawn error: {e}\n")
                rec["status"], rec["rc"] = "failed", -1
        rec["wall_s"] = round(time.perf_counter() - t0, 2)
        after = snapshot_results()
        rec["outputs"] = sorted(p for p in after
                                if p not in before
                                or after[p] > before[p])
        print(f"[{rec['status']:>5s}] {job['tag']} "
              f"{rec['wall_s']}s -> {log.name}", flush=True)

    # ---- schedule: submit ready jobs, wait, repeat until drained
    scheduled = set()
    while True:
        for job in jobs:
            t = job["tag"]
            if t in scheduled or records[t]["status"] != "pending":
                continue
            if stop["flag"]:
                continue                     # drain: no new submissions
            if not deps_done(job):
                continue                     # wait for in-matrix deps
            deps = missing_deps(job)
            if deps:
                log = logs / f"{t}.log"
                log.write_text(f"# {t}\nSKIPPED (missing deps): "
                               f"{'; '.join(deps)}\n")
                records[t].update(status="skipped_dep", rc=None,
                                  log=str(log))
                print(f"[skip] {t}: missing {deps}", flush=True)
                scheduled.add(t)
                continue
            fut = pools[job["kind"]].submit(run_job, job)
            futures[fut] = t
            scheduled.add(t)
        if not futures:
            break                            # nothing running, nothing
                                              # left to submit
        done, _ = wait(list(futures), timeout=0.5,
                       return_when=FIRST_COMPLETED)
        for f in done:
            futures.pop(f)
        if stop["flag"] and not futures:
            for job in jobs:                 # mark unsubmitted as drained
                t = job["tag"]
                if records[t]["status"] == "pending":
                    records[t]["status"] = "skipped_sigint"
    for p in pools.values():
        p.shutdown(wait=True)

    counts = {}
    for r in records.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    manifest = args.manifest or str(
        RESULTS / f"parallel_manifest_{time.strftime('%Y%m%d_%H%M%S')}.json")
    Path(manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest).write_text(json.dumps(
        {"counts": counts, "n_jobs": len(records),
         "cpu_workers": args.cpu_workers, "gpu_workers": args.gpu_workers,
         "smoke": args.smoke, "jobs": list(records.values())}, indent=2))
    print(f"[run_parallel] {counts} -> {manifest}")


if __name__ == "__main__":
    main()
