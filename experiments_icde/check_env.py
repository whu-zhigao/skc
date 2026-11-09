#!/usr/bin/env python3
"""Pre-flight environment check for the ICDE experiment suite.

Run this BEFORE any experiment, especially on a fresh GPU node with
zero datasets present (that is the whole point of this script):

    python3 check_env.py            # full FOUND/MISSING table + verdict
    python3 check_env.py --quiet    # one line, for scripting
    echo $?                         # 0 iff READY

Checks, in order:
  1. Python >= 3.9.
  2. numpy and torch importable (torch CUDA availability is INFO only;
     the suite degrades to CPU-class stages without CUDA).
  3. CODE_ROOT as auto-resolved by common._find_code_root(), plus the
     SKC_CODE_ROOT override if set.
  4. The mplignn package (mplignn.data / mplignn.model) importable.
  5. Every dataset .pkl the full paper matrix needs (diginetica,
     gowalla), from DATA_ROOT = SKC_DATA_ROOT or CODE_ROOT/data.
  6. The checkpoints b4/b7 need, using the SAME resolvers as
     run_parallel.py: NEW = abl_no_residual_<ds>.pt or mplignn_<ds>.pt;
     OLD = env OLD_CKPT (with '{ds}' template) or the per-dataset
     default pair abl_no_anchor_<ds>.pt / abl_nodenoise_<ds>.pt.
  7. INFO-only extras: faiss (d1 baselines), systems.bench_retrieval
     (reference neighbor heuristic; an exact-Jaccard fallback exists),
     and any existing a2 drift-trace npz files.

Every MISSING row prints the exact fix command. Nothing is written and
no experiment is executed."""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

MIN_PY = (3, 9)
REQUIRED_DATASETS = ("diginetica", "gowalla")
INFO_DATASETS = ("yoochoose_quarter",)      # d2 stage (not in the
                                             # diginetica/gowalla matrix)

FIX_CODE_ROOT = ("place code/ (mplignn/, data/) next to experiments_icde/, "
                 "or export SKC_CODE_ROOT=/path/to/code")
FIX_DATA = ("copy <dataset>.pkl from the repo's code/data/ into {root}, "
            "or export SKC_DATA_ROOT=/dir/with/pkl/files")
FIX_CKPT = ("copy {path} from the repo's code/results/real/, or pass a "
            "--ckpt flag / export OLD_CKPT")
FIX_TORCH = "python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu121"
FIX_NUMPY = "python3 -m pip install numpy"
FIX_MPLIGNN = "copy the repo's code/mplignn/ next to experiments_icde/"

ROWS: list = []          # (name, status, detail, fix)


def row(name, status, detail="", fix=""):
    ROWS.append((name, status, detail, fix))


def check_python():
    v = sys.version_info[:3]
    if sys.version_info >= MIN_PY:
        row("python", "FOUND", f"{v[0]}.{v[1]}.{v[2]}")
    else:
        row("python", "MISSING", f"{v[0]}.{v[1]}.{v[2]} < {MIN_PY[0]}.{MIN_PY[1]}",
            f"install/activate Python >= {MIN_PY[0]}.{MIN_PY[1]}")


def check_numpy_torch():
    try:
        np = importlib.import_module("numpy")
        row("numpy", "FOUND", np.__version__)
    except Exception as e:                                # noqa: BLE001
        row("numpy", "MISSING", str(e), FIX_NUMPY)
    try:
        th = importlib.import_module("torch")
        cuda = th.cuda.is_available()
        row("torch", "FOUND", f"{th.__version__} "
            f"(cuda_available={cuda}, device_count={th.cuda.device_count()})")
        if not cuda:
            row("torch-cuda", "INFO", "not available; suite will run "
                "CPU-class stages (GPU stages a2-inline/b2/b7 degrade "
                "to CPU or stay queued)", FIX_TORCH)
    except Exception as e:                                # noqa: BLE001
        row("torch", "MISSING", str(e), FIX_TORCH)


def check_root_and_data():
    """CODE_ROOT resolution + mplignn + datasets + ckpts."""
    suite = Path(__file__).resolve().parent
    sys.path.insert(0, str(suite))
    try:
        import common                                     # noqa: F401
    except Exception as e:                                # noqa: BLE001
        row("common.py import", "MISSING", str(e),
            "numpy/torch must be importable first (see rows above)")
        return
    env_root = os.environ.get("SKC_CODE_ROOT", "")
    matched = [str(common.CODE)]
    detail = (f"resolved={common.CODE}"
              + (f"  [SKC_CODE_ROOT={env_root}]" if env_root else ""))
    if (common.CODE / "mplignn").is_dir():
        row("CODE_ROOT (mplignn/)", "FOUND", detail)
    else:
        row("CODE_ROOT (mplignn/)", "MISSING", detail, FIX_CODE_ROOT)
        return
    try:
        import mplignn.data                               # noqa: F401
        import mplignn.model                              # noqa: F401
        row("mplignn package", "FOUND", str(Path(mplignn.__file__).parent))
    except Exception as e:                                # noqa: BLE001
        row("mplignn package", "MISSING", str(e), FIX_MPLIGNN)
        _ = matched

    env_data = os.environ.get("SKC_DATA_ROOT", "")
    root = common.DATA_ROOT
    for ds in REQUIRED_DATASETS:
        p = common.data_file(ds)
        if p.exists():
            row(f"dataset {ds}.pkl", "FOUND", str(p))
        else:
            row(f"dataset {ds}.pkl", "MISSING", str(p),
                FIX_DATA.format(root=root))
    for ds in INFO_DATASETS:
        p = common.data_file(ds)
        row(f"dataset {ds}.pkl (optional)", "INFO",
            f"{p} -- {'present' if p.exists() else 'absent'} "
            "(not in the default matrix)")
    if env_data:
        row("SKC_DATA_ROOT", "INFO", f"{env_data} (data dir override active)")

    # checkpoints -- reuse the run_parallel resolvers verbatim
    try:
        from run_parallel import _default_ckpt, _old_ckpt
    except Exception as e:                                # noqa: BLE001
        row("run_parallel import", "MISSING", str(e),
            "suite files incomplete; re-copy experiments_icde/")
        return
    for ds in REQUIRED_DATASETS:
        new = Path(_default_ckpt(ds))
        if new.exists():
            row(f"ckpt NEW {ds}", "FOUND", str(new))
        else:
            row(f"ckpt NEW {ds}", "MISSING", str(new),
                FIX_CKPT.format(path=new))
        old = _old_ckpt(ds)
        if old and Path(old).exists():
            note = "  [from env OLD_CKPT]" if os.environ.get("OLD_CKPT") \
                else "  [per-dataset default]"
            row(f"ckpt OLD {ds} (b4/b7)", "FOUND", str(old) + note)
        elif old:
            row(f"ckpt OLD {ds} (b4/b7)", "MISSING", str(old),
                FIX_CKPT.format(path=old))
        else:
            cand = common.REAL / f"abl_no_anchor_{ds}.pt"
            row(f"ckpt OLD {ds} (b4/b7)", "MISSING",
                f"no OLD_CKPT env, no abl_no_anchor_{ds}.pt / "
                f"abl_nodenoise_{ds}.pt in {common.REAL}",
                f"export OLD_CKPT=<earlier-generation .pt> or copy "
                f"{cand}")
    for extra, label in (("systems.bench_retrieval",
                          "reference neighbor heuristic"),
                         ("faiss", "d1 ANN baselines")):
        try:
            importlib.import_module(extra)
            row(f"optional: {extra}", "INFO", f"present ({label})")
        except Exception:                                 # noqa: BLE001
            row(f"optional: {extra}", "INFO",
                f"absent ({label}; fallback/skip path is used)")
    # d2 (Yoochoose-1/4 scaling study): pkl above + its own ckpt
    # (resolver chain mirrors exp_d2_yoochoose.resolve_ckpt).
    for ds in INFO_DATASETS:
        ck = None
        for name in (f"abl_no_residual_{ds}.pt", f"mplignn_{ds}.pt",
                     f"full_mplignn_{ds}.pt"):
            cand = common.REAL / name
            if cand.exists():
                ck = cand
                break
        row(f"ckpt {ds} (d2, optional)", "INFO",
            f"{ck or common.REAL / f'mplignn_{ds}.pt'} -- "
            f"{'present' if ck else 'absent'} (d2 ann section skips "
            "gracefully when missing)")
    for npz in sorted(common.RESULTS.glob("drift_trace_*.npz")):
        row("a2 drift trace (generated)", "INFO", str(npz))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quiet", action="store_true",
                    help="print only the verdict line (for scripts)")
    args = ap.parse_args()

    check_python()
    check_numpy_torch()
    check_root_and_data()

    missing = [r for r in ROWS if r[1] == "MISSING"]
    ready = not missing
    verdict = "READY" if ready else "NOT-READY"
    if args.quiet:
        print(verdict + ("" if ready
                         else " (" + "; ".join(m[0] for m in missing) + ")"))
        return 0 if ready else 1

    w = max(len(r[0]) for r in ROWS) if ROWS else 4
    print(f"{'CHECK':<{w}}  STATUS   DETAIL")
    print("-" * (w + 60))
    for name, status, detail, fix in ROWS:
        print(f"{name:<{w}}  {status:<8s} {detail}")
        if status == "MISSING" and fix:
            print(f"{'':<{w}}           fix: {fix}")
    print("-" * (w + 60))
    print(f"VERDICT: {verdict}"
          + ("" if ready
             else "  -- fix every MISSING row above and re-run"))
    print("next: python3 run_parallel.py --dry-run   "
          "(lists the job matrix + per-job deps)")
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
