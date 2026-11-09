#!/usr/bin/env python3
"""C2: gate-sparsity spectrum and Tier-3 score cache. Measures
exact-zero rates of {softmax, tanh, entmax15} over Q K^T from the
model's embedding table (auto-probe; random W_q/W_k fallback flagged in
JSON), PLUS the two model gate SITES (gate_sites: anchor/SAtt vs
denoise/NAtt entmax zero rates from a real checkpoint batch), derives
budgets B=(1-p)N^2*4B, simulates LFU pair admission, and times
cached-vs-recompute Tier-3 matmuls on GPU."""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import config_out, load_data, load_model, timeit, write_json


def entmax15(X, n_iter=30):
    """entmax alpha=1.5 via bisection (Peters et al., 2019)."""
    import torch
    max_val, _ = X.max(dim=-1, keepdim=True)
    lo = max_val - 1.0
    hi = max_val - 1.0 / X.shape[-1]
    for _ in range(n_iter):
        tau = (lo + hi) / 2.0
        s = torch.clamp(X - tau, min=0.0).sum(dim=-1, keepdim=True)
        lo = torch.where(s >= 1.0, tau, lo)
        hi = torch.where(s < 1.0, tau, hi)
    return torch.clamp(X - (lo + hi) / 2.0, min=0.0)


def zero_rate(fn, Q, K, chunk=2048):
    tot = zeros = 0
    for i in range(0, Q.shape[0], chunk):
        S = fn(Q[i:i + chunk] @ K.T)
        zeros += int((S == 0).sum().item())
        tot += S.numel()
    return zeros / tot


def gate_sites(model, data, dev, k=10, batch=16):
    """Exact-zero rates of the model's entmax gates at the TWO gate
    SITES of RelationEnhancement: the anchor-augmented SAtt stream
    (model.py Eq. 6-8; paper row 'alpha-EntMax (anchor)') and the
    denoising NAtt cross-attention (Eq. 9-10; paper row 'alpha-EntMax
    (denoise)'). One real batch goes through the model; the site score
    tensors are rebuilt from the model's own tare weights and passed
    through tare._sparse_attn (the model's own gate, including its
    softmax fallback at alpha<=1.06), so the measurement matches what
    the checkpoint actually computes."""
    import torch
    from mplignn.data import build_adjacency
    from common import neighbor_ids, pool_profiles
    if getattr(model, "no_denoise", False):
        return "skipped: no_denoise architecture (tare not called)"
    tare = model.tare
    captured = {}

    def pre_hook(module, args):
        captured.setdefault("args", args)

    hook = tare.register_forward_pre_hook(pre_hook)
    try:
        with torch.no_grad():
            sess = [s if len(s) > 1 else list(s) + list(s)
                    for s in data.train[:batch]]
            B = len(sess)
            L = max(len(s) for s in sess)
            seqs = torch.zeros(B, L, dtype=torch.long)
            for i, s in enumerate(sess):
                seqs[i, :len(s)] = torch.tensor(s)
            lens = (seqs != 0).sum(dim=1).clamp(min=1).long()
            a = build_adjacency(seqs, lens)
            nbrs = neighbor_ids(sess, data.train, k)
            emb_np = model.embedding.weight.detach().cpu().numpy()
            r0 = torch.from_numpy(pool_profiles(
                emb_np, [data.train[int(i)] for nb in nbrs for i in nb]
            ).reshape(B, k, emb_np.shape[1])).float().to(dev)
            a_r = (torch.eye(k, device=dev)
                   .unsqueeze(0).expand(B, k, k).contiguous())
            model(seqs.to(dev), lens.to(dev), a.to(dev), a_r, r0)
    finally:
        hook.remove()

    with torch.no_grad():
        x_loc, glob_map, a_r, r0 = captured["args"]
        B = x_loc.shape[0]
        # relation smoothing (Eq. 9) replicated on the captured inputs
        r = tare.rel_proj(r0)
        deg = a_r.sum(-1, keepdim=True).clamp_min(1.0)
        for _ in range(tare.rel_layers):
            r = torch.bmm(a_r / deg, r) + r / 2.0
        r_bar = r.mean(dim=1)
        # ---- site 1: anchor-augmented SAtt stream (Eq. 6-8)
        h_hat = (torch.cat([glob_map, tare.anchor.view(1, -1)], dim=0)
                 if tare.use_anchor else glob_map)
        n_s = h_hat.shape[0]
        q = torch.relu(tare.w_q1(h_hat))
        qh = q.view(n_s, tare.h, tare.dk).transpose(0, 1)
        kh = h_hat.view(n_s, tare.h, tare.dk).transpose(0, 1)
        S1 = qh @ kh.transpose(-2, -1) / (2 * tare.dk) ** 0.5
        alpha_row = tare._adaptive_alpha(h_hat[-1, :].unsqueeze(0))
        attn1 = tare._sparse_attn(S1.unsqueeze(0), alpha_row).squeeze(0)
        # ---- site 2: denoising NAtt cross-attention (Eq. 10)
        q_n = torch.relu(tare.w_qn(x_loc))
        qh2 = q_n.view(B, -1, tare.h, tare.dk).transpose(1, 2)
        rh = r.view(B, -1, tare.h, tare.dk).transpose(1, 2)
        S2 = qh2 @ rh.transpose(-2, -1) / (2 * tare.dk) ** 0.5
        alpha2 = tare._adaptive_alpha_n(r_bar)
        attn2 = tare._sparse_attn(S2, alpha2)
    return {
        "batch": B, "k_neighbors": k,
        "anchor_satt": {
            "alpha": float(alpha_row.float().mean()),
            "softmax_fallback": bool(alpha_row.mean() <= 1.06),
            "exact_zero_rate": float((attn1 == 0).float().mean()),
            "n_scores": int(attn1.numel())},
        "denoise_natt": {
            "alpha": float(alpha2.float().mean()),
            "softmax_fallback_frac": float(
                (alpha2 <= 1.06).float().mean()),
            "exact_zero_rate": float((attn2 == 0).float().mean()),
            "n_scores": int(attn2.numel())},
        "probe": "one real batch through model.tare (exact top-k "
                 "neighbor profiles as r0, identity a_r); site scores "
                 "rebuilt from the checkpoint's tare weights per "
                 "model.py Eq. 6-10; gate = tare._sparse_attn (the "
                 "model's own entmax with softmax fallback at "
                 "alpha<=1.06)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--out", default=None,
                    help="legacy single-file mode (always rewritten)")
    ap.add_argument("--force", action="store_true",
                    help="rerun even if the per-config JSON exists")
    args = ap.parse_args()

    stem = args.data.stem
    ck_tag = (f"_{args.ckpt.stem}" if args.ckpt else "_defaultckpt")
    out_path, skip = config_out("c2", f"{stem}{ck_tag}", args.out,
                                args.force)
    if skip:
        print(f"[c2] exists, skipping: {out_path} (--force to rerun)")
        return

    import torch
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_data(args.data)
    model = load_model(data, args.ckpt, device=dev)
    E = model.embedding.weight.detach().to(dev).float()
    N, d = E.shape

    names = dict(model.named_parameters())
    cand = [n for n in names if "vocab" in n.lower()
            and any(t in n.lower() for t in ("query", "key", ".q_", ".k_"))]
    probe = "embedding+random-projection(seed=0)"
    g = torch.Generator(device="cpu").manual_seed(0)
    Wq = torch.randn(d, d, generator=g).to(dev) / np.sqrt(d)
    Wk = torch.randn(d, d, generator=g).to(dev) / np.sqrt(d)
    if len(cand) >= 2:
        probe = f"model params: {cand[:2]}"
        Wq = names[cand[0]].detach().to(dev).float().reshape(-1, d)[:, :d]
        Wk = names[cand[1]].detach().to(dev).float().reshape(-1, d)[:, :d]
    Q, K = E @ Wq.T, E @ Wk.T

    gates = {"softmax": lambda S: F.softmax(S, dim=-1),
             "tanh": torch.tanh, "entmax15": entmax15}
    spectrum = []
    for name, fn in gates.items():
        p = zero_rate(fn, Q, K)
        spectrum.append({"gate": name, "exact_zero_rate": p,
                         "support_fraction": 1.0 - p,
                         "budget_mb_fp32": (1.0 - p) * N * N * 4 / 1e6,
                         "feasible_under_1gb":
                             (1.0 - p) * N * N * 4 <= 1e9})
        print(f"  {name}: p={p:.4f} "
              f"budget={(1.0 - p) * N * N * 4 / 1e6:.1f}MB")

    sites = gate_sites(model, data, dev)
    if isinstance(sites, dict):
        print(f"  gate site anchor/satt: alpha="
              f"{sites['anchor_satt']['alpha']:.3f} zero="
              f"{sites['anchor_satt']['exact_zero_rate']:.4f}")
        print(f"  gate site denoise/natt: alpha="
              f"{sites['denoise_natt']['alpha']:.3f} zero="
              f"{sites['denoise_natt']['exact_zero_rate']:.4f}")
    else:
        print(f"  gate sites: {sites}")

    pairs = Counter()
    for s in data.train[:50000]:
        for a, b in zip(s[:-1], s[1:]):
            pairs[(min(a, b), max(a, b))] += 1
    stream = list(pairs.elements())
    rng = np.random.default_rng(0)
    rng.shuffle(stream)
    probe_n = min(len(stream), 100000)
    lfu_rows = []
    for frac in (0.01, 0.05, 0.1, 0.25, 0.5):
        cap = max(1, int(len(pairs) * frac))
        top = {p_ for p_, _ in pairs.most_common(cap)}
        hit = sum(1 for p_ in stream[:probe_n] if p_ in top) / probe_n
        lfu_rows.append({"budget_frac": frac, "hit": hit})

    S_dense = torch.tanh(Q[:2048] @ K.T)
    n_rows = S_dense.shape[0]   # gather probes the materialized slice
    idx = torch.randint(0, n_rows, (65536,), device=dev)
    t_cache = timeit(lambda: S_dense[idx], 10, 100)
    t_recomp = timeit(lambda: torch.tanh(Q[:2048] @ K.T), 10, 100)

    write_json(out_path, {
        "dataset": stem, "n_items": N, "probe": probe,
        "spectrum": spectrum, "gate_sites": sites,
        "full_precompute_gb": N * N * 4 / 1e9,
        "lfu_pairs": lfu_rows,
        "tier3_latency": {"cached_gather": t_cache,
                          "dense_recompute_2048rows": t_recomp}})


if __name__ == "__main__":
    main()
