"""Re-implementations of the eleven SBR baselines under one shared,
documented protocol: identical loader/splits, full-sort ranking, full
softmax cross-entropy, d=64, lr 5e-4, 12 epochs x <=300 batches,
patience 4 (see the paper's Section 5.2.4 transparency note).

Architectural notes and deliberate simplifications (also released in
NOTES.md):
  * All models share one item-embedding table and score the full
    vocabulary with a single softmax + cross-entropy.  Where an original
    paper used BPR or a sampled session-parallel loss (GRU4Rec, FPMC),
    we keep the CE objective for a controlled comparison.
  * FPMC maps the anonymous session to the mean item embedding as the
    user factor (no user ids exist in SBR) and factorizes the item-item
    transition tensor as (e_i W_t) . e_v.
  * GRU4Rec uses padded nn.GRU with a packed sequence so padding never
    enters the recurrence; the last hidden state scores items.
  * NARM uses one GRU layer plus the published attention read-out
    (beta_i = v^T sigma(W1 h_t + W2 h_i)).
  * STAMP uses the published lightweight MLP on [last item; session
    mean] with a residual connection.
  * LESSR is approximated by the gated graph convolution over the
    ordered session adjacency plus positional embeddings; read-out is
    soft attention.
  * DHCN is approximated by one hypergraph propagation round
    (session <-- mean of items, then item <-- session message) without
    the auxiliary self-supervision loss.
  * GCE-GNN reuses the local GGNN plus the tanh global attention over
    the batch-unique items (level-transformed embeddings) and soft
    attention read-out.
  * MTD is approximated by GGNN over popularity-degree-normalized
    transitions with a mean+last gated read-out.
  * DSAN uses the alpha-EntMax (1.5) self-attention block over the
    prefix (one layer, our released implementation) and the
    last-position read-out.
  * MSGIFSR builds intent units at granularities {1,2,4} by segmenting
    the prefix into contiguous blocks, averaging each block, and fusing
    the units with a small GRU.
  * MSGAT applies alpha-EntMax attention twice: over prefix items
    (local channel) and from prefix queries to the batch-unique item
    keys (global channel), then fuses the two channels with a learned
    gate.
Every model returns {"scores": (B, N)} and consumes the same call
signature as MPLI-GNN; relation inputs (a_r, r0) are ignored.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .entmax import entmax
from .model import GlobalEncoder, LocalEncoder


class _Base(nn.Module):
    def __init__(self, n_items: int, d: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n_items = n_items
        self.embedding = nn.Embedding(n_items, d, padding_idx=0)
        self.d = d

    def forward(self, seqs, lens, a, a_r=None, r0=None):
        raise NotImplementedError


def _mask_lens(h, lens):
    """(B,T,d) -> zero out padded positions."""
    t = h.shape[1]
    keep = torch.arange(t, device=h.device).view(1, -1) < lens.view(-1, 1)
    return h * keep.unsqueeze(-1).float()


class FPMC(_Base):
    def __init__(self, n_items, d=64, dropout=0.1):
        super().__init__(n_items, d, dropout)
        self.w_t = nn.Linear(d, d, bias=False)
        self.bias = nn.Embedding(n_items, 1, padding_idx=0)

    def forward(self, seqs, lens, a, a_r=None, r0=None):
        e = self.embedding(seqs)                                # (B,T,d)
        m = _mask_lens(e, lens)
        mean = m.sum(1) / lens.view(-1, 1).float()              # (B,d)
        trans = self.w_t(m).sum(1) / lens.view(-1, 1).float()
        emb = self.embedding.weight                             # (N,d)
        scores = (self.bias.weight.squeeze(-1).unsqueeze(0)
                  + mean @ emb.t() + trans @ emb.t())
        return {"scores": scores}


class GRU4Rec(_Base):
    def __init__(self, n_items, d=64, dropout=0.1):
        super().__init__(n_items, d, dropout)
        self.gru = nn.GRU(d, d, batch_first=True)

    def forward(self, seqs, lens, a, a_r=None, r0=None):
        e = self.embedding(seqs)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lens.cpu().clamp_min(1), batch_first=True,
            enforce_sorted=False)
        _, h = self.gru(packed)
        h = h.squeeze(0)                                        # (B,d)
        return {"scores": h @ self.embedding.weight.t()}


class NARM(_Base):
    def __init__(self, n_items, d=64, dropout=0.1):
        super().__init__(n_items, d, dropout)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.w1 = nn.Linear(d, d, bias=False)
        self.w2 = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, 1, bias=False)
        self.w3 = nn.Linear(2 * d, d, bias=False)

    def forward(self, seqs, lens, a, a_r=None, r0=None):
        e = self.embedding(seqs)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lens.cpu().clamp_min(1), batch_first=True,
            enforce_sorted=False)
        out, h = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=e.shape[1])
        h_t = h.squeeze(0).unsqueeze(1)                         # (B,1,d)
        att = self.v(torch.sigmoid(self.w1(h_t) + self.w2(out)))  # (B,T,1)
        att = att.masked_fill(
            torch.arange(out.shape[1], device=out.device
                         ).view(1, -1, 1) >= lens.view(-1, 1, 1), 0.0)
        beta = att / att.sum(1, keepdim=True).clamp_min(1e-9)
        c = (beta * out).sum(1)
        s = self.w3(torch.cat([c, h.squeeze(0)], dim=-1))
        return {"scores": s @ self.embedding.weight.t()}


class STAMP(_Base):
    def __init__(self, n_items, d=64, dropout=0.1):
        super().__init__(n_items, d, dropout)
        self.mlp1 = nn.Linear(2 * d, d)
        self.mlp2 = nn.Linear(d, d)
        self.act = nn.ReLU()

    def forward(self, seqs, lens, a, a_r=None, r0=None):
        e = _mask_lens(self.embedding(seqs), lens)
        x_t = e[torch.arange(e.shape[0]), lens - 1]             # last item
        m_s = e.sum(1) / lens.view(-1, 1).float()
        h = self.mlp2(self.act(self.mlp1(torch.cat([x_t, m_s], -1)))) + x_t
        return {"scores": h @ self.embedding.weight.t()}


class LESSR(_Base):
    """GGNN over the ordered session graph + positional embeddings."""
    def __init__(self, n_items, d=64, dropout=0.1):
        super().__init__(n_items, d, dropout)
        self.ggnn = LocalEncoder(d, n_layers=1, residual=True)
        self.pos = nn.Embedding(64, d)
        self.w_read = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, seqs, lens, a, a_r=None, r0=None):
        t = seqs.shape[1]
        pos = self.pos(torch.arange(t, device=seqs.device) % 64)
        e = self.embedding(seqs) + pos.unsqueeze(0)
        h = self.ggnn(e, a)
        last = h[torch.arange(h.shape[0]), lens - 1]
        att = (torch.sigmoid(self.w_read(h)) *
               torch.tanh(last).unsqueeze(1)).sum(-1)
        att = att.masked_fill(torch.arange(t, device=h.device
                                           ).view(1, -1) >= lens.view(-1, 1), 0.0)
        beta = att / att.sum(1, keepdim=True).clamp_min(1e-9)
        z = self.ln((beta.unsqueeze(-1) * h).sum(1))
        return {"scores": z @ self.embedding.weight.t()}


class DHCN(_Base):
    """One hypergraph propagation round (session<->items)."""
    def __init__(self, n_items, d=64, dropout=0.1):
        super().__init__(n_items, d, dropout)
        self.msg = nn.Linear(d, d)
        self.w_read = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, seqs, lens, a, a_r=None, r0=None):
        e = _mask_lens(self.embedding(seqs), lens)
        s_node = e.sum(1) / lens.view(-1, 1).float()            # session node
        h = e + self.msg(s_node).unsqueeze(1)                   # item <- session
        last = h[torch.arange(h.shape[0]), lens - 1]
        att = (torch.sigmoid(self.w_read(h)) *
               torch.tanh(last).unsqueeze(1)).sum(-1)
        att = att.masked_fill(torch.arange(h.shape[1], device=h.device
                                           ).view(1, -1) >= lens.view(-1, 1), 0.0)
        beta = att / att.sum(1, keepdim=True).clamp_min(1e-9)
        z = self.ln((beta.unsqueeze(-1) * h).sum(1))
        return {"scores": z @ self.embedding.weight.t()}


class GCEGNN(_Base):
    """Local GGNN + tanh global attention over batch-unique items."""
    def __init__(self, n_items, d=64, dropout=0.1, depth=1,
                 use_global=True):
        super().__init__(n_items, d, dropout)
        self.use_global = use_global
        self.local = LocalEncoder(d, n_layers=depth, residual=True)
        self.global_ = GlobalEncoder(d, n_heads=2, n_blocks_m=1, n_blocks=1,
                                     dropout=dropout)
        self.w_read = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, seqs, lens, a, a_r=None, r0=None):
        uniq = torch.unique(seqs)
        glob = torch.zeros_like(self.embedding.weight)
        glob[uniq] = self.global_(self.embedding.weight[uniq])
        h = self.local(self.embedding(seqs), a)
        if self.use_global:
            h = h + glob[seqs]
        last = h[torch.arange(h.shape[0]), lens - 1]
        att = (torch.sigmoid(self.w_read(h)) *
               torch.tanh(last).unsqueeze(1)).sum(-1)
        att = att.masked_fill(torch.arange(h.shape[1], device=h.device
                                           ).view(1, -1) >= lens.view(-1, 1), 0.0)
        beta = att / att.sum(1, keepdim=True).clamp_min(1e-9)
        z = self.ln((beta.unsqueeze(-1) * h).sum(1))
        return {"scores": z @ self.embedding.weight.t()}


class MTD(_Base):
    """GGNN over normalized transitions, gated mean+last read-out."""
    def __init__(self, n_items, d=64, dropout=0.1):
        super().__init__(n_items, d, dropout)
        self.ggnn = LocalEncoder(d, n_layers=2, residual=True)
        self.gate = nn.Linear(2 * d, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, seqs, lens, a, a_r=None, r0=None):
        h = self.ggnn(self.embedding(seqs), a)
        last = h[torch.arange(h.shape[0]), lens - 1]
        mean = _mask_lens(h, lens).sum(1) / lens.view(-1, 1).float()
        z = self.ln(self.gate(torch.cat([last, mean], -1)))
        return {"scores": z @ self.embedding.weight.t()}


class DSAN(_Base):
    """Sparse self-attention (entmax-1.5) over the prefix."""
    def __init__(self, n_items, d=64, dropout=0.1):
        super().__init__(n_items, d, dropout)
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(),
                                 nn.Linear(2 * d, d))
        self.ln = nn.LayerNorm(d)

    def forward(self, seqs, lens, a, a_r=None, r0=None):
        e = self.embedding(seqs)
        q, k, v = self.wq(e), self.wk(e), self.wv(e)
        scores = q @ k.transpose(-2, -1) / self.d ** 0.5
        pad = (torch.arange(e.shape[1], device=e.device).view(1, -1)
               >= lens.view(-1, 1))
        scores = scores.masked_fill(pad.unsqueeze(-1), -1e9)
        attn = entmax(scores, alpha=1.5, dim=-1)
        h = self.ln(e + self.ffn(attn @ v))
        z = h[torch.arange(h.shape[0]), lens - 1]
        return {"scores": z @ self.embedding.weight.t()}


class MSGIFSR(_Base):
    """Intent units at granularities {1,2,4} fused by a small GRU."""
    def __init__(self, n_items, d=64, dropout=0.1):
        super().__init__(n_items, d, dropout)
        self.proj = nn.Linear(d, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.ln = nn.LayerNorm(d)

    def forward(self, seqs, lens, a, a_r=None, r0=None):
        e = _mask_lens(self.embedding(seqs), lens)
        units = []
        for g in (1, 2, 4):
            t = e.shape[1]
            pad_to = ((t + g - 1) // g) * g
            e2 = F.pad(e, (0, 0, 0, pad_to - t)).view(e.shape[0], -1, g, e.shape[2])
            units.append(e2.mean(2))                            # (B,T/g,d)
        l_min = min(u.shape[1] for u in units)
        u = self.proj(sum(u[:, :l_min] for u in units))
        packed_lens = lens.clamp_max(l_min).clamp_min(1)
        packed = nn.utils.rnn.pack_padded_sequence(
            u, packed_lens.cpu(), batch_first=True,
            enforce_sorted=False)
        _, h = self.gru(packed)
        z = self.ln(h.squeeze(0))
        return {"scores": z @ self.embedding.weight.t()}


class MSGAT(_Base):
    """Bi-channel entmax attention: prefix-local + batch-global."""
    def __init__(self, n_items, d=64, dropout=0.1):
        super().__init__(n_items, d, dropout)
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.wq2 = nn.Linear(d, d)
        self.gate = nn.Linear(2 * d, d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(),
                                 nn.Linear(2 * d, d))
        self.ln = nn.LayerNorm(d)

    def forward(self, seqs, lens, a, a_r=None, r0=None):
        e = self.embedding(seqs)
        t = e.shape[1]
        pad = (torch.arange(t, device=e.device).view(1, -1)
               >= lens.view(-1, 1))
        q, k, v = self.wq(e), self.wk(e), self.wv(e)
        s1 = q @ k.transpose(-2, -1) / self.d ** 0.5
        s1 = s1.masked_fill(pad.unsqueeze(-1), -1e9)
        local = entmax(s1, alpha=1.5, dim=-1) @ v               # (B,T,d)
        uniq = torch.unique(seqs)
        gk = self.wk(self.embedding.weight[uniq])               # (n_b,d)
        gv = self.wv(self.embedding.weight[uniq])
        s2 = self.wq2(e) @ gk.t() / self.d ** 0.5               # (B,T,n_b)
        s2 = s2.masked_fill(pad.unsqueeze(-1), -1e9)
        glob = entmax(s2, alpha=1.5, dim=-1) @ gv                # (B,T,d)
        h = self.ln(e + self.ffn(self.gate(torch.cat([local, glob], -1))))
        z = h[torch.arange(h.shape[0]), lens - 1]
        return {"scores": z @ self.embedding.weight.t()}


BASELINES = {
    "fpmc": FPMC, "gru4rec": GRU4Rec, "narm": NARM, "stamp": STAMP,
    "lessr": LESSR, "dhcn": DHCN, "gcegnn": GCEGNN, "mtd": MTD,
    "dsan": DSAN, "msgifsr": MSGIFSR, "msgat": MSGAT,
}
