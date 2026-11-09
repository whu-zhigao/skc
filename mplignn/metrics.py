"""Full-sort ranking metrics (Section 5.2.2)."""
from __future__ import annotations

import torch

__all__ = ["full_sort_metrics"]


@torch.no_grad()
def full_sort_metrics(scores: torch.Tensor, targets: torch.Tensor,
                      ks=(10, 20)) -> dict:
    """Compute HR@k, MRR@k, NDCG@k and Coverage@k under full sort.

    scores: (B, N) full-vocabulary scores; targets: (B,) ground truth ids.
    Coverage counts distinct items appearing in each user's top-k.
    """
    out = {}
    b, n = scores.shape
    top20 = scores.topk(20, dim=-1).indices                    # (B, 20)
    hit20 = top20 == targets.view(-1, 1)
    for k in ks:
        hit = hit20[:, :k]
        rank = hit.float().argmax(dim=-1) + 1.0                # 1-based if hit
        found = hit.any(dim=-1)
        out[f"HR@{k}"] = hit.any(dim=-1).float().mean().item()
        out[f"MRR@{k}"] = (1.0 / rank * found.float()).sum().item() / b
        disc = 1.0 / torch.log2(rank + 1.0)                    # rank 1 -> 1.0
        dcg = (disc * found.float()).sum().item()
        out[f"NDCG@{k}"] = dcg / b
        out[f"Coverage@{k}"] = top20[:, :k].unique().numel() / n
    return out
