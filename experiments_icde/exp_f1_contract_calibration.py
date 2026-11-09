#!/usr/bin/env python3
"""F1: calibrate bounded dispersion against end-to-end ranking quality.

The probe controls per-item embedding drift, then mixes fresh and stale
neighbor profiles within requests.  Unlike the earlier two-checkpoint probe,
it reports quality as a surface over item drift and stale fraction, and
records realized profile dispersion rather than inferring it from epoch ids.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (RESULTS, load_data, load_model, neighbor_ids,  # noqa: E402
                    pool_profiles, run_forward, serving_prefixes,
                    write_json)
from mplignn.data import pad_batch  # noqa: E402


def perturb_embeddings(emb: np.ndarray, item_drift: float,
                       seed: int) -> np.ndarray:
    """Return a table whose every row moves by exactly item_drift L2."""
    if item_drift == 0:
        return emb.copy()
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=emb.shape).astype(np.float32)
    direction /= np.maximum(
        np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    return emb + item_drift * direction


@torch.no_grad()
def evaluate_profiles(model, sessions, neighbors, profiles, device,
                      batch_size: int):
    """Batched next-item evaluation with explicit query prefixes."""
    hits20 = 0
    reciprocal20 = []
    for start in range(0, len(sessions), batch_size):
        chunk = sessions[start:start + batch_size]
        neighbor_batch = neighbors[start:start + len(chunk)]
        profile_batch = profiles[np.asarray(neighbor_batch, dtype=np.int64)]
        seqs, lens, _ = pad_batch(chunk, device)
        profiles_t = torch.from_numpy(profile_batch).to(device)
        out = run_forward(model, seqs, profiles_t)
        logits = out.float()
        targets = torch.tensor([session[-1] for session in chunk],
                               device=logits.device, dtype=torch.long)
        ranks = (logits.argsort(dim=1, descending=True)
                 == targets[:, None]).nonzero(as_tuple=False)[:, 1] + 1
        hits20 += int((ranks <= 20).sum().item())
        reciprocal20.extend((1.0 / ranks[ranks <= 20]).cpu().tolist())
    return {
        "HR@20": hits20 / len(sessions),
        "MRR@20": float(np.mean(reciprocal20)) if reciprocal20 else 0.0,
    }


def dispersion_stats(fresh: np.ndarray, stale: np.ndarray,
                     ids: np.ndarray) -> dict:
    delta = np.linalg.norm(stale[ids] - fresh[ids], axis=1)
    return {
        "profile_mean": float(delta.mean()),
        "profile_p50": float(np.percentile(delta, 50)),
        "profile_p95": float(np.percentile(delta, 95)),
        "profile_p99": float(np.percentile(delta, 99)),
        "profile_max": float(delta.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--n-query", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--drifts", default="0,0.05,0.10,0.20,0.40,0.80")
    parser.add_argument("--fractions", default="0,0.25,0.50,0.75,1.0")
    parser.add_argument("--seeds", default="7,17")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_data(args.data)
    model = load_model(data, args.ckpt, device=device)
    emb = model.embedding.weight.detach().cpu().numpy()
    sessions = data.test[:min(args.n_query, len(data.test))]
    prefixes = serving_prefixes(sessions)
    neighbors = np.asarray(neighbor_ids(prefixes, data.train, 10),
                           dtype=np.int64)
    fresh_profiles = pool_profiles(emb, data.train)
    active_ids = np.unique(neighbors.reshape(-1))

    drifts = [float(x) for x in args.drifts.split(",")]
    fractions = [float(x) for x in args.fractions.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    points = []
    started = time.perf_counter()

    for seed in seeds:
        for drift in drifts:
            stale_emb = perturb_embeddings(emb, drift, seed)
            stale_profiles = pool_profiles(stale_emb, data.train)
            stats = dispersion_stats(fresh_profiles, stale_profiles,
                                     active_ids)
            for fraction in fractions:
                mixed = fresh_profiles.copy()
                count = int(round(fraction * neighbors.shape[1]))
                if count:
                    choice = np.random.default_rng(1000 + seed).permuted(
                        np.tile(np.arange(neighbors.shape[1]),
                                (len(neighbors), 1)), axis=1)[:, :count]
                    rows, cols = np.meshgrid(np.arange(len(neighbors)),
                                             np.arange(count), indexing="ij")
                    selected = neighbors[rows, choice]
                    mixed[selected] = stale_profiles[selected]
                metrics = evaluate_profiles(
                    model, sessions, neighbors, mixed, device,
                    args.batch_size)
                points.append({
                    "seed": seed,
                    "item_drift": drift,
                    "stale_fraction": fraction,
                    **metrics,
                    **stats,
                })
                print(f"seed={seed} drift={drift:.2f} "
                      f"f={fraction:.2f}: {metrics['HR@20']:.3f} "
                      f"disp={stats['profile_p99']:.3f}", flush=True)

    result = {
        "dataset": args.data.stem,
        "checkpoint": None if args.ckpt is None else args.ckpt.name,
        "device": device,
        "n_queries": len(sessions),
        "k_neighbors": neighbors.shape[1],
        "protocol": ("prefix-only query; exact per-item L2 drift; "
                     "random neighbor-row generation mixing"),
        "drifts": drifts,
        "fractions": fractions,
        "seeds": seeds,
        "elapsed_seconds": time.perf_counter() - started,
        "points": points,
    }
    out = args.out or (RESULTS / "f1" /
                       f"contract_calibration_{args.data.stem}.json")
    write_json(out, result)


if __name__ == "__main__":
    main()
