"""Sparse attention mappings for MPLI-GNN.

Implements the alpha-EntMax family used by the target-anchored relation
enhancement module (Section 4.4 of the paper):

    EntMax_alpha(z)_i = max(0, (alpha-1) * (z_i - tau)) ** (1/(alpha-1))

where tau is chosen so that the output sums to one.  alpha -> 1 recovers
softmax, alpha = 2 recovers sparsemax, and alpha = 1.5 is the classic
entmax-1.5 of Peters et al. (2019).  tau is found by bisection and
detached, so gradients flow through the piecewise-linear support, the
standard treatment in sparse-attention implementations.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["entmax", "sparsemax", "adaptive_entmax"]


def _tsallis_map(z: torch.Tensor, tau: torch.Tensor, alpha: float) -> torch.Tensor:
    """Map scores through the Tsallis link, zero below threshold."""
    inner = (alpha - 1.0) * (z - tau)
    return inner.clamp_min(0.0).pow(1.0 / (alpha - 1.0))


def entmax(z: torch.Tensor, alpha: float = 1.5, dim: int = -1,
           n_iter: int = 40) -> torch.Tensor:
    """Differentiable alpha-EntMax over dimension ``dim``.

    Args:
        z: score tensor of shape (..., n).
        alpha: sparsity temperature in (1, 2].
        dim: reduction dimension.
        n_iter: bisection iterations (40 gives <1e-6 mass error).
    """
    if alpha <= 1.0 + 1e-6:  # softmax limit
        return F.softmax(z, dim=dim)
    if abs(alpha - 2.0) < 1e-6:  # sparsemax has a closed form
        return sparsemax(z, dim=dim)

    z_star = z.transpose(dim, -1) if dim != -1 else z
    z_max = z_star.amax(dim=-1, keepdim=True)
    z_min = z_star.amin(dim=-1, keepdim=True)
    # Lower bound: only the largest score active with mass 1
    # (alpha-1)(z_max - tau) = 1  =>  tau_lo = z_max - 1/(alpha-1)
    tau_lo = z_max - 1.0 / (alpha - 1.0)
    # Upper bound: everything below threshold -> mass 0
    tau_hi = z_max.expand_as(z_star).clone()
    # Guard degenerate rows (all equal scores) by widening the interval
    tau_lo = torch.minimum(tau_lo, z_min - 1.0)

    for _ in range(n_iter):
        tau = 0.5 * (tau_lo + tau_hi)
        mass = _tsallis_map(z_star, tau, alpha).sum(dim=-1, keepdim=True)
        too_big = mass > 1.0
        # larger tau -> smaller mass
        tau_lo = torch.where(too_big, tau, tau_lo)
        tau_hi = torch.where(too_big, tau_hi, tau)

    tau = 0.5 * (tau_lo + tau_hi).detach()
    p = _tsallis_map(z_star, tau, alpha)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    if dim != -1:
        p = p.transpose(dim, -1)
    return p


def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Closed-form sparsemax (alpha = 2)."""
    z_star = z.transpose(dim, -1) if dim != -1 else z
    sorted_z, _ = z_star.sort(dim=-1, descending=True)
    k = torch.arange(1, z_star.shape[-1] + 1, device=z.device, dtype=z.dtype)
    cumsum = sorted_z.cumsum(dim=-1)
    support = sorted_z * k > (cumsum - 1.0)
    k_support = support.sum(dim=-1, keepdim=True).float().clamp_min(1.0)
    tau = (cumsum.gather(-1, k_support.long() - 1) - 1.0) / k_support
    p = (z_star - tau).clamp_min(0.0)
    if dim != -1:
        p = p.transpose(dim, -1)
    return p


def adaptive_entmax(scores: torch.Tensor, alpha: torch.Tensor, dim: int = -1,
                    n_iter: int = 40) -> torch.Tensor:
    """Apply EntMax with a per-row adaptive alpha in (1, 2].

    ``alpha`` has shape (..., 1).  When alpha is very close to 1 we fall
    back to softmax for numerical stability; the bisection otherwise
    handles the full range.  Batching over distinct alphas is done by
    grouping rows, which keeps the routine exact.
    """
    a = alpha.squeeze(-1)
    out = torch.empty_like(scores)
    softmax_rows = a <= 1.0 + 1e-3
    if softmax_rows.any():
        out[softmax_rows] = F.softmax(scores[softmax_rows], dim=dim)
    rest = ~softmax_rows
    if rest.any():
        levels = a[rest].unique()
        for lv in levels.tolist():
            mask = rest & (a == lv)
            out[mask] = entmax(scores[mask], alpha=lv, dim=dim, n_iter=n_iter)
    return out
