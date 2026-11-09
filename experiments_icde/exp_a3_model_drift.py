#!/usr/bin/env python3
"""A3: embedding-drift snapshots from the actual MPLI co-training loop.

A2 uses an embedding-plus-linear-head surrogate for inexpensive long traces.
A3 complements it with the actual recommendation model: each mini-batch uses
neighbor profiles from the live embedding table, backpropagates next-item
cross-entropy, and snapshots the embedding table at the requested cadence.
The output has the same ``drift`` shape expected by DAI-2 replay and F6.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (REAL, config_out, load_data, load_model,  # noqa: E402
                    run_forward, write_json)
from mplignn.data import pad_batch  # noqa: E402


def default_ckpt(dataset: str) -> Path:
    for name in (
        f"abl_no_residual_{dataset}.pt",
        f"mplignn_{dataset}.pt",
    ):
        path = REAL / name
        if path.exists():
            return path
    raise FileNotFoundError(f"no checkpoint for {dataset}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps-per-epoch", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--neighbor-pool", type=int, default=8192,
                    help="fixed profile pool used by the cached co-training arm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-model", type=Path, default=None,
                    help="optional checkpoint path for the final co-trained model")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import torch

    stem = args.data.stem
    ckpt = args.ckpt or default_ckpt(stem)
    tag = f"{stem}_mpli_e{args.epochs}_s{args.steps_per_epoch}_seed{args.seed}"
    out_path, skip = config_out("a3", tag, args.out, args.force)
    if skip:
        print(f"[a3] exists, skipping: {out_path} (--force to rerun)")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.data)
    model = load_model(data, ckpt, device=str(device))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # The older B7 co-training protocol fixes the cached neighbor set at the
    # start of training.  Preserve that behavior here while avoiding an exact
    # Jaccard search for every mini-batch.  A uniformly sampled pool makes the
    # actual-model drift trace inexpensive while neighbor profiles remain
    # differentiable functions of the live embedding table.
    pool_size = min(args.neighbor_pool, len(data.train))
    neighbor_pool = np.sort(rng.choice(
        len(data.train), size=pool_size, replace=False
    )).astype(np.int64)
    pool_item_lists = [
        sorted(set(data.train[int(sid)])) for sid in neighbor_pool.tolist()
    ]
    pool_width = max((len(x) for x in pool_item_lists), default=1)
    pool_width = max(pool_width, 1)
    pool_item_ids = np.zeros(
        (len(neighbor_pool), pool_width), dtype=np.int64
    )
    pool_item_mask = np.zeros_like(pool_item_ids, dtype=np.float32)
    for row, items in enumerate(pool_item_lists):
        if items:
            pool_item_ids[row, :len(items)] = items
            pool_item_mask[row, :len(items)] = 1.0
    pool_item_tensor = torch.from_numpy(pool_item_ids).to(device)
    pool_mask_tensor = torch.from_numpy(pool_item_mask).to(device)

    prev = model.embedding.weight.detach().cpu().numpy().copy()
    rows = []
    snapshots = []
    losses = []
    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        for _step in range(args.steps_per_epoch):
            idx = rng.integers(0, len(data.train), size=args.batch_size)
            batch = [data.train[int(i)] for i in idx]
            pool_rows = rng.integers(
                0, len(neighbor_pool), size=(len(batch), args.k)
            )
            flat_pool_rows = torch.from_numpy(
                pool_rows.reshape(-1)
            ).to(device)
            item_idx = pool_item_tensor[flat_pool_rows]
            item_mask = pool_mask_tensor[flat_pool_rows]
            profile_rows = (
                model.embedding.weight[item_idx]
                * item_mask.unsqueeze(-1)
            ).sum(dim=1) / item_mask.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
            profiles = profile_rows.view(len(batch), args.k, -1)
            seqs, _lens, targets = pad_batch(batch, str(device))
            logits = run_forward(
                model,
                seqs,
                profiles,
            ).float()
            loss = torch.nn.functional.cross_entropy(
                logits, targets.to(device)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach()))

        current = model.embedding.weight.detach().cpu().numpy().copy()
        drift = np.linalg.norm(current - prev, axis=1)
        snapshots.append(drift)
        rows.append({
            "epoch": epoch,
            "mean_loss": float(np.mean(epoch_losses)),
            "drift_mean": float(drift.mean()),
            "drift_p50": float(np.percentile(drift, 50)),
            "drift_p90": float(np.percentile(drift, 90)),
            "drift_p99": float(np.percentile(drift, 99)),
            "drift_max": float(drift.max()),
        })
        print(
            f"epoch={epoch} loss={rows[-1]['mean_loss']:.4f} "
            f"drift_p99={rows[-1]['drift_p99']:.6f}", flush=True
        )
        prev = current
        losses.extend(epoch_losses)

    trace = np.stack(snapshots).astype(np.float64)
    result = {
        "dataset": stem,
        "checkpoint": str(ckpt),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0)
        if device.type == "cuda" else None,
        "model": "MPLI_GNN",
        "protocol": (
            "actual-model co-training; every batch recomputes neighbor "
            "profiles from the live embedding table before forward"
        ),
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "neighbor_pool_size": int(pool_size),
        "epochs_data": rows,
        "trace_path": str(out_path.with_suffix(".npz")),
        "saved_model": str(args.save_model) if args.save_model else None,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path.with_suffix(".npz"), drift=trace)
    if args.save_model is not None:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_model)
    write_json(out_path, result)


if __name__ == "__main__":
    main()
