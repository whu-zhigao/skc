#!/usr/bin/env python3
"""A2: REAL per-item embedding drift traces (replaces the randn
simulation in the old bench). Modes:
  --ckpts DIR : per-item drift between consecutive checkpoints
                (real co-training snapshots; preferred).
  default     : inline training of embedding table + linear head with
                real next-item cross-entropy on the real corpus (real
                optimizer, real gradients), snapshotting the table.
Output: results/systems/drift_trace_<stem>.npz (T, n_items) drift
norms + stats JSON whose quantiles choose DAI thetas."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import config_out, ensure_results, write_json


def load_emb(p: Path) -> np.ndarray:
    import torch
    obj = torch.load(p, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        if "embedding.weight" in obj:
            return obj["embedding.weight"].float().numpy()
        for v in obj.values():
            if hasattr(v, "dim") and v.dim() == 2:
                return v.float().numpy()
        raise ValueError(f"{p}: no embedding tensor found")
    return obj.float().numpy()


def inline_train(data, d=64, steps=2000, snap_every=50, bs=512,
                 lr=1e-3, seed=0):
    """Next-item CE on the pooled prefix: h = masked mean of item
    embeddings; predict the last item. Real gradients move exactly the
    embedding rows DAI reasons about."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    emb = torch.nn.Embedding(data.n_items, d).to(dev)
    head = torch.nn.Linear(d, data.n_items).to(dev)
    opt = torch.optim.AdamW(
        list(emb.parameters()) + list(head.parameters()), lr=lr)
    rng = np.random.default_rng(seed)
    sessions = [s if len(s) > 1 else s + s for s in data.train]
    snaps, prev = [], emb.weight.detach().clone()
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(sessions), bs)
        batch = [sessions[i] for i in idx]
        L = max(len(s) for s in batch)
        seqs = torch.zeros(bs, L - 1, dtype=torch.long, device=dev)
        mask = torch.zeros(bs, L - 1, device=dev)
        tgt = torch.zeros(bs, dtype=torch.long, device=dev)
        for b, s in enumerate(batch):
            seqs[b, :len(s) - 1] = torch.tensor(s[:-1])
            mask[b, :len(s) - 1] = 1.0
            tgt[b] = s[-1]
        h = (emb(seqs) * mask.unsqueeze(-1)).sum(1) / \
            mask.sum(1, keepdim=True).clamp(min=1.0)
        loss = F.cross_entropy(head(h), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % snap_every == 0 or step == 1:
            w = emb.weight.detach()
            snaps.append((w.cpu().numpy(), prev.cpu().numpy()))
            prev = w.clone()
            print(f"  step {step}: loss {loss.item():.4f}")
    return snaps  # list of (new, old)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpts", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--snap-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="legacy single-file mode (always rewritten)")
    ap.add_argument("--force", action="store_true",
                    help="rerun even if the per-config JSON exists")
    args = ap.parse_args()
    stem = args.data.stem
    mode = "ckpts" if args.ckpts is not None else "inline"
    npz = ensure_results() / f"drift_trace_{stem}.npz"
    tag = (f"{stem}_{mode}" if mode == "ckpts"
           else f"{stem}_inline_steps{args.steps}_s{args.seed}")
    out_path, skip = config_out("a2", tag, args.out, args.force)
    # inline mode must also leave the npz for B3/C1; require both on resume
    if skip and (mode == "ckpts" or npz.exists()):
        print(f"[a2] exists, skipping: {out_path} (--force to rerun)")
        return
    skip = False

    if args.ckpts is not None:
        ck = sorted(args.ckpts.glob("*.pt"))
        assert len(ck) >= 2, "need >=2 checkpoints"
        snaps = [load_emb(p) for p in ck]
        drift = np.stack([np.linalg.norm(snaps[i + 1] - snaps[i], axis=1)
                          for i in range(len(snaps) - 1)])
    else:
        from common import load_data
        data = load_data(args.data)
        pairs = inline_train(data, steps=args.steps,
                             snap_every=args.snap_every,
                             seed=args.seed)
        drift = np.stack([np.linalg.norm(new - old, axis=1)
                          for new, old in pairs])

    q = {f"q{p}": float(np.percentile(drift, p))
         for p in (10, 25, 50, 75, 90, 95, 99)}
    ensure_results()
    np.savez_compressed(npz, drift=drift, mode=mode)
    write_json(out_path, {
        "dataset": stem, "mode": mode,
        "n_intervals": drift.shape[0], "n_items": drift.shape[1],
        "drift_quantiles": q,
        "frac_items_moving": {
            f"q{p}": float((drift > np.percentile(drift, p)).mean())
            for p in (50, 75, 90, 95)},
        "note": "DAI thetas should be chosen from these quantiles"})


if __name__ == "__main__":
    main()
