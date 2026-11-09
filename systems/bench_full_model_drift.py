#!/usr/bin/env python3
"""Full-model drift audit for embedding and attention projection weights.

A3 established embedding drift.  This experiment extends the actual MPLI
co-training loop to all parameters and reports whether an embedding-only page
ledger can cover Tier-2 projections.  The intended outcome is diagnostic, not
a faster training loop: whenever projection weights move, the derived-view
definition changes and active projection rows must be invalidated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments_icde.common import (  # noqa: E402
    load_data, load_model, run_forward,
)
from mplignn.data import pad_batch  # noqa: E402


PROJECTION_KEYS = (
    "global_.wq.weight", "global_.wk.weight", "global_.wv.weight",
    "global_.wo.weight", "tare.w_qn.weight", "tare.w_n.weight",
    "intent.w_q.weight", "intent.w_k.weight",
)


def tensor_delta(old, new):
    old = old.detach().cpu()
    new = new.detach().cpu()
    delta = new - old
    return {
        "l2": float(torch.linalg.vector_norm(delta)),
        "relative": float(
            torch.linalg.vector_norm(delta)
            / torch.linalg.vector_norm(old).clamp_min(1e-12)
        ),
        "max_abs": float(delta.abs().max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--neighbor-pool", type=int, default=8192)
    ap.add_argument("--projection-sample", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.data)
    model = load_model(data, args.ckpt, device=str(device))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    pool_size = min(args.neighbor_pool, len(data.train))
    neighbor_pool = np.sort(rng.choice(len(data.train), pool_size, replace=False)).astype(np.int64)
    pool_items = [sorted(set(data.train[int(sid)])) for sid in neighbor_pool]
    width = max(max(map(len, pool_items)), 1)
    ids = np.zeros((pool_size, width), dtype=np.int64)
    mask = np.zeros_like(ids, dtype=np.float32)
    for row, items in enumerate(pool_items):
        ids[row, :len(items)] = items
        mask[row, :len(items)] = 1
    pool_ids = torch.from_numpy(ids).to(device)
    pool_mask = torch.from_numpy(mask).to(device)

    sample_items = torch.from_numpy(
        np.sort(rng.choice(data.n_items, size=min(args.projection_sample, data.n_items), replace=False))
    ).long().to(device)
    prev_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    prev_embedding = model.embedding.weight.detach().cpu().numpy().copy()
    embedding_trace = []
    rows = []
    losses = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        epoch_losses = []
        for _step in range(args.steps_per_epoch):
            batch_idx = rng.integers(0, len(data.train), size=args.batch_size)
            batch = [data.train[int(i)] for i in batch_idx]
            chosen = rng.integers(0, pool_size, size=(len(batch), 10))
            flat = torch.from_numpy(chosen.reshape(-1)).to(device)
            item_idx = pool_ids[flat]
            item_mask = pool_mask[flat]
            profiles = (
                model.embedding.weight[item_idx] * item_mask.unsqueeze(-1)
            ).sum(dim=1) / item_mask.sum(dim=1, keepdim=True).clamp_min(1)
            profiles = profiles.view(len(batch), 10, -1)
            seqs, _lens, targets = pad_batch(batch, str(device))
            logits = run_forward(model, seqs, profiles).float()
            loss = torch.nn.functional.cross_entropy(logits, targets.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach()))

        current_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        current_embedding = model.embedding.weight.detach().cpu().numpy()
        embedding_drift = np.linalg.norm(current_embedding - prev_embedding, axis=1)
        embedding_trace.append(embedding_drift)

        parameter_deltas = {
            k: tensor_delta(prev_state[k], current_state[k])
            for k in current_state
            if current_state[k].numel() > 1
        }
        projection_rows = {}
        with torch.no_grad():
            for key in PROJECTION_KEYS:
                if key not in current_state:
                    continue
                w_old = prev_state[key].to(device).float()
                w_new = current_state[key].to(device).float()
                e_old = torch.from_numpy(prev_embedding).to(device).float()[sample_items]
                e_new = model.embedding.weight[sample_items].float()
                cached = w_old @ e_old.T
                fresh = w_new @ e_new.T
                actual = (fresh - cached).norm(dim=0)
                embedding_only_bound = (
                    torch.linalg.matrix_norm(w_new, ord=2)
                    * (e_new - e_old).norm(dim=1)
                )
                projection_rows[key] = {
                    **parameter_deltas[key],
                    "actual_projection_error_mean": float(actual.mean()),
                    "actual_projection_error_p99": float(actual.quantile(0.99)),
                    "embedding_only_bound_mean": float(embedding_only_bound.mean()),
                    "embedding_only_bound_violations": int(
                        (actual > embedding_only_bound + 1e-6).sum()
                    ),
                }

        all_projection_deltas = [parameter_deltas[k]["l2"] for k in PROJECTION_KEYS if k in parameter_deltas]
        row = {
            "epoch": epoch,
            "mean_loss": float(np.mean(epoch_losses)),
            "seconds": time.perf_counter() - epoch_start,
            "embedding_drift_mean": float(embedding_drift.mean()),
            "embedding_drift_p50": float(np.percentile(embedding_drift, 50)),
            "embedding_drift_p99": float(np.percentile(embedding_drift, 99)),
            "embedding_drift_max": float(embedding_drift.max()),
            "all_parameter_l2": float(np.linalg.norm([
                x["l2"] for x in parameter_deltas.values()
            ])),
            "projection_weight_l2": float(np.linalg.norm(all_projection_deltas)),
            "projection_view_changed": bool(max(all_projection_deltas, default=0.0) > 0),
            "projections": projection_rows,
        }
        rows.append(row)
        print(
            f"epoch={epoch} loss={row['mean_loss']:.4f} "
            f"emb_p99={row['embedding_drift_p99']:.6f} "
            f"proj_l2={row['projection_weight_l2']:.6f}", flush=True
        )
        prev_state = current_state
        prev_embedding = current_embedding.copy()
        losses.extend(epoch_losses)

    stem = args.data.stem
    out = args.out or (
        ROOT / "results" / "systems" / "full_model_drift" /
        f"{stem}_seed{args.seed}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    trace = np.stack(embedding_trace).astype(np.float64)
    np.savez_compressed(out.with_suffix(".npz"), drift=trace)
    result = {
        "dataset": stem,
        "checkpoint": str(args.ckpt),
        "seed": args.seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "batch_size": args.batch_size,
        "neighbor_pool_size": pool_size,
        "projection_sample_size": int(sample_items.numel()),
        "protocol": "actual MPLI AdamW co-training; full state-dict delta; projection cache old-W/old-E versus current-W/current-E",
        "epochs_data": rows,
        "summary": {
            "epochs_with_projection_view_change": sum(r["projection_view_changed"] for r in rows),
            "embedding_drift_p99_mean": float(np.mean([r["embedding_drift_p99"] for r in rows])),
            "projection_weight_l2_mean": float(np.mean([r["projection_weight_l2"] for r in rows])),
        },
    }
    out.write_text(json.dumps(result, indent=2))
    print(out)


if __name__ == "__main__":
    main()
