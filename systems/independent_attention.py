"""An attention consumer independent of MPLI-GNN's denoising block."""
from __future__ import annotations

import torch
from torch import nn


class IndependentCrossAttentionConsumer(nn.Module):
    """Standard scaled dot-product cross-attention over neighbor profiles.

    The item table is initialized from an existing checkpoint but frozen. The
    attention, output projection, and layer norm are trained from scratch;
    therefore the consumer is not a variant of MPLI-GNN's RelationEnhancement.
    """
    def __init__(self, item_embeddings: torch.Tensor, dimension: int = 64,
                 heads: int = 4):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            item_embeddings.clone().float(), freeze=True, padding_idx=0)
        self.query = nn.Sequential(nn.LayerNorm(dimension),
                                   nn.Linear(dimension, dimension))
        self.attention = nn.MultiheadAttention(dimension, heads,
                                                batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(dimension),
                                 nn.Linear(dimension, dimension),
                                 nn.Tanh())

    def forward(self, seqs: torch.Tensor, lens: torch.Tensor,
                neighbor_profiles: torch.Tensor):
        mask = (torch.arange(seqs.shape[1], device=seqs.device)[None, :]
                < lens[:, None])
        items = self.embedding(seqs) * mask.unsqueeze(-1).to(
            self.embedding.weight.dtype)
        query = self.query(items.sum(1) /
                           lens[:, None].clamp(min=1).to(items.dtype))
        context, _ = self.attention(query.unsqueeze(1),
                                    neighbor_profiles,
                                    neighbor_profiles,
                                    need_weights=False)
        representation = self.out(context.squeeze(1))
        return representation @ self.embedding.weight.t()
