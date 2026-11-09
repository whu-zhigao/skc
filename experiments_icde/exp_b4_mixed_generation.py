#!/usr/bin/env python3
"""B4: mixed-generation consistency boundary. Evaluates HR@20/MRR@10
when neighbor profiles come from: fresh table, stale single-generation
table, or a fraction-f mix. Validated/falsifies Prop. (mixed-generation
degradation). Needs --old-ckpt (an earlier-generation checkpoint)."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (config_out, load_data, load_model, neighbor_ids,
                    pool_profiles, ranking_metrics, run_forward,
                    serving_prefixes, write_json)


def eval_with_profiles(model, data, nbrs, P, dev, n_items):
    import torch
    scores, targets = [], []
    for i, s in enumerate(data.test[:len(nbrs)]):
        s = s if len(s) > 1 else s + s
        seq = torch.tensor([s[:-1]], device=dev)
        prof = torch.from_numpy(P[nbrs[i]]).unsqueeze(0).to(dev)
        out = run_forward(model, seq, prof)
        logits = out[0] if isinstance(out, tuple) else out
        logits = logits.reshape(-1).float()
        if logits.numel() >= n_items:
            scores.append(logits[:n_items].cpu().numpy())
        targets.append(s[-1])
    return ranking_metrics(np.stack(scores), np.array(targets), ks=(20,))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--old-ckpt", type=Path, required=True)
    ap.add_argument("--n-query", type=int, default=1000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default=None,
                    help="legacy single-file mode (always rewritten)")
    ap.add_argument("--force", action="store_true",
                    help="rerun even if the per-config JSON exists")
    args = ap.parse_args()

    stem = args.data.stem
    out_path, skip = config_out("b4", f"{stem}_nq{args.n_query}_mixedgen",
                                args.out, args.force)
    if skip:
        print(f"[b4] exists, skipping: {out_path} (--force to rerun)")
        return

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_data(args.data)
    model = load_model(data, args.ckpt, device=dev)
    emb_new = model.embedding.weight.detach().cpu().numpy()
    emb_old = torch.load(args.old_ckpt, map_location="cpu",
                         weights_only=True)["embedding.weight"].numpy()

    nq = min(args.n_query, len(data.test))
    nbrs = [np.asarray(nb) for nb in
            neighbor_ids(serving_prefixes(data.test[:nq]), data.train, args.k)]
    P_new = pool_profiles(emb_new, data.train)
    P_old = pool_profiles(emb_old, data.train)
    sigma = float(np.linalg.norm(emb_new - emb_old)
                  / np.linalg.norm(emb_old))

    rng = np.random.default_rng(0)
    out = {"dataset": stem, "sigma_frobenius": sigma, "points": []}
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        P = P_new.copy()
        if f > 0:
            for i, nb in enumerate(nbrs):
                m = int(round(f * len(nb)))
                if m:
                    swap = rng.choice(len(nb), m, replace=False)
                    P[nb[swap]] = P_old[nb[swap]]
        try:
            m = eval_with_profiles(model, data, nbrs, P, dev,
                                   data.n_items)
            out["points"].append({"stale_fraction": f, **m})
            print(f"  f={f:.2f}: {m}")
        except RuntimeError as e:
            print(f"[WARN] eval skipped at f={f}: {e}")
            out["points"].append({"stale_fraction": f, "skipped": str(e)})
    write_json(out_path, out)


if __name__ == "__main__":
    main()
