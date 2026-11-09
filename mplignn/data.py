"""Data pipeline for MPLI-GNN.

Provides (i) session preprocessing with the standard SBR filtering rules,
(ii) local adjacency construction for a batch of padded sessions,
(iii) a synthetic benchmark with *planted* interest shifts so that the
denoising mechanism can be validated against ground truth, and (iv) the
session-relation retrieval used by the global memory.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = ["SessionData", "SyntheticConfig", "generate_synthetic",
           "save_sessions", "load_sessions", "preprocess",
           "build_adjacency", "SessionRelationIndex", "pad_batch",
           "relation_matrix"]


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------
@dataclass
class SessionData:
    train: List[List[int]]
    valid: List[List[int]]
    test: List[List[int]]
    n_items: int
    shift_flags: Optional[np.ndarray] = None  # ground-truth shift labels (synthetic)


@dataclass
class SyntheticConfig:
    n_items: int = 600
    n_groups: int = 20          # latent intent groups
    n_train: int = 4000
    n_valid: int = 400
    n_test: int = 800
    min_len: int = 3
    max_len: int = 12
    shift_prob: float = 0.25    # probability a session contains a planted shift
    shift_frac: float = 0.34    # fraction of tail clicks drawn from another group
    seed: int = 7


# ---------------------------------------------------------------------------
# Synthetic benchmark with planted interest shifts
# ---------------------------------------------------------------------------
def generate_synthetic(cfg: SyntheticConfig) -> SessionData:
    """Generate sessions from latent intent groups with optional shifts.

    Each session draws a latent group; most items come from that group.
    With probability ``shift_prob`` the last ``shift_frac`` of the session
    is drawn from a different group, mimicking a discount-driven drift.
    The returned ``shift_flags`` mark which test sessions contain a shift,
    giving RQ5's stratified analysis ground truth on synthetic data.
    """
    rng = np.random.default_rng(cfg.seed)
    items_per_group = np.array_split(np.arange(cfg.n_items), cfg.n_groups)

    def draw_session(with_shift: bool) -> Tuple[List[int], bool]:
        length = int(rng.integers(cfg.min_len, cfg.max_len + 1))
        g = int(rng.integers(cfg.n_groups))
        g2 = int(rng.integers(cfg.n_groups - 1))
        g2 = g2 + 1 if g2 >= g else g2
        pool = items_per_group[g]
        pool2 = items_per_group[g2]
        n_shift = max(1, int(round(cfg.shift_frac * length))) if with_shift else 0
        seq: List[int] = []
        for k in range(length - n_shift):
            seq.append(int(rng.choice(pool)))
        for k in range(n_shift):
            seq.append(int(rng.choice(pool2)))
        return seq, bool(n_shift > 0)

    def make_split(n: int) -> Tuple[List[List[int]], np.ndarray]:
        seqs, flags = [], []
        for _ in range(n):
            shifted = bool(rng.random() < cfg.shift_prob)
            s, f = draw_session(shifted)
            seqs.append(s)
            flags.append(f)
        return seqs, np.array(flags, dtype=bool)

    train, _ = make_split(cfg.n_train)
    valid, _ = make_split(cfg.n_valid)
    test, tflags = make_split(cfg.n_test)
    return SessionData(train=train, valid=valid, test=test,
                       n_items=cfg.n_items, shift_flags=tflags)


def save_sessions(data: SessionData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_sessions(path: Path) -> SessionData:
    with open(path, "rb") as f:
        return pickle.load(f)


def preprocess(data: SessionData, min_item_freq: int = 5) -> SessionData:
    """Standard SBR filtering: drop length-1 sessions and rare items."""
    from collections import Counter
    freq = Counter(i for s in data.train for i in s)
    keep = {i for i, c in freq.items() if c >= min_item_freq}
    remap = {i: j for j, i in enumerate(sorted(keep))}

    def _f(seqs: List[List[int]]) -> List[List[int]]:
        out = []
        for s in seqs:
            s2 = [remap[i] for i in s if i in keep]
            if len(s2) >= 2:
                out.append(s2)
        return out

    return SessionData(train=_f(data.train), valid=_f(data.valid),
                       test=_f(data.test), n_items=len(remap),
                       shift_flags=data.shift_flags)


# ---------------------------------------------------------------------------
# Batching and local graphs
# ---------------------------------------------------------------------------
def pad_batch(sessions: Sequence[Sequence[int]], device: str = "cpu"
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a list of sessions into next-item training/eval instances.

    Returns:
        seqs:    (B, T) int64 item ids of the *prefix* s[:-1], 0-padded.
        lens:    (B,) int64 true prefix lengths (len(s) - 1 >= 1).
        targets: (B,) int64 last item of each session (never in seqs).
    """
    lens = torch.tensor([max(len(s) - 1, 1) for s in sessions],
                        dtype=torch.long)
    t_max = int(lens.max().item())
    seqs = torch.zeros(len(sessions), t_max, dtype=torch.long)
    targets = torch.zeros(len(sessions), dtype=torch.long)
    for b, s in enumerate(sessions):
        prefix = s[:-1] if len(s) >= 2 else s
        seqs[b, :len(prefix)] = torch.tensor(prefix, dtype=torch.long)
        targets[b] = s[-1]
    return seqs.to(device), lens.to(device), targets.to(device)


def build_adjacency(seqs: torch.Tensor, lens: torch.Tensor,
                    add_reverse: bool = True) -> torch.Tensor:
    """Row-normalized adjacency over session positions, shape (B, T, T).

    Nodes are the T sequence positions of the padded session (position p
    carries the embedding of the p-th clicked item).  Consecutive
    positions are connected; repeated item ids therefore keep separate
    nodes, which preserves temporal multiplicity.  A^s and its transpose
    encode out/in edges as in SR-GNN.  Rows are normalized with
    Equation (1) of the paper.
    """
    b, t = seqs.shape
    a = torch.zeros(b, t, t, device=seqs.device)
    if t > 1:
        idx = torch.arange(t - 1, device=seqs.device)
        a[:, idx, idx + 1] += 1.0
        if add_reverse:
            a[:, idx + 1, idx] += 1.0
    # mask padded positions (rows and columns beyond each true length)
    pos = torch.arange(t, device=seqs.device)
    pad = pos.view(1, -1) >= lens.view(-1, 1)                # (B, T)
    a = a.masked_fill(pad.view(b, t, 1), 0.0)
    a = a.masked_fill(pad.view(b, 1, t), 0.0)
    deg = a.sum(dim=-1, keepdim=True)
    return a / deg.clamp_min(1.0)


# ---------------------------------------------------------------------------
# Session-relation retrieval (global memory initialisation)
# ---------------------------------------------------------------------------
class SessionRelationIndex:
    """Top-K_r cosine-similar training sessions for the relation GCN.

    Training sessions are embedded by their normalized item histograms;
    retrieval is exact matrix multiplication, replaceable by an ANN index
    at Web scale.
    """

    def __init__(self, train: List[List[int]], n_items: int, k_r: int = 10):
        self.k_r = k_r
        self.n_items = n_items
        mat = np.zeros((len(train), n_items), dtype=np.float32)
        for r, s in enumerate(train):
            for i in s:
                mat[r, i] += 1.0
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self.emb = mat / np.clip(norms, 1e-9, None)
        self.train = train

    def retrieve(self, session: Sequence[int]) -> List[List[int]]:
        q = np.zeros(self.n_items, dtype=np.float32)
        for i in session:
            q[i] += 1.0
        q /= max(float(np.linalg.norm(q)), 1e-9)
        sims = self.emb @ q
        top = np.argpartition(-sims, self.k_r)[:self.k_r]
        return [self.train[r] for r in top]


class FastRelationIndex:
    """Precomputed top-K_r neighbors with batch-time tensor lookup.

    Neighbors are found once per dataset with an inverted index over
    items: for each session, candidates share its rarest item and are
    ranked by shared-item count (Jaccard-style co-click similarity).
    This is the index-accelerated realization of the cosine retrieval in
    Section 3.4, cached to disk so repeated runs are free.

    At batch time the relation inputs are assembled with tensor ops:
        r0[b, j] = mean embedding of neighbor j's items
        a_r       = row-normalized cosine of the r0 profiles
    """

    def __init__(self, anchor_sessions: List[List[int]],
                 corpus: List[List[int]], n_items: int, k_r: int = 10,
                 l_cap: int = 12, max_cand_per_item: int = 64,
                 cache: Optional[Path] = None, keep_corpus: bool = False):
        self.k_r, self.l_cap, self.n_items = k_r, l_cap, n_items
        # Revision (M3 controls): optionally retain the raw corpus rows so
        # batch_relation() can draw uniformly random training sessions
        # (control="rand").  Opt-in: only the training-side index needs it,
        # and Yoochoose-scale runs must not grow their memory footprint.
        self.corpus_items: Optional[torch.Tensor] = None
        self.n_corpus = len(corpus)
        if cache is not None and Path(cache).exists():
            blob = np.load(Path(cache))
            self.neigh_items = torch.from_numpy(blob["items"])
            self.anchor_ids = torch.arange(len(anchor_sessions))
            if keep_corpus:
                ci = np.zeros((self.n_corpus, l_cap), dtype=np.int64)
                for idx, s in enumerate(corpus):
                    ci[idx, :min(len(s), l_cap)] = s[:l_cap]
                self.corpus_items = torch.from_numpy(ci)
            return
        m, n = len(anchor_sessions), len(corpus)
        inv: Dict[int, List[int]] = {}
        for idx, s in enumerate(corpus):
            for i in set(s):
                inv.setdefault(i, []).append(idx)
        freq = {i: len(v) for i, v in inv.items()}

        corpus_items = np.zeros((n, l_cap), dtype=np.int64)
        for idx, s in enumerate(corpus):
            corpus_items[idx, :min(len(s), l_cap)] = s[:l_cap]

        neigh = np.zeros((m, k_r), dtype=np.int64)
        for idx, s in enumerate(anchor_sessions):
            # candidates from the rarest item of the session
            items = sorted(set(s), key=lambda i: freq.get(i, 1 << 30))
            cands: Dict[int, int] = {}
            for it in items[:3]:
                for j in inv.get(it, [])[:max_cand_per_item]:
                    if j != idx:
                        cands[j] = cands.get(j, 0) + 1
            top = sorted(cands.items(), key=lambda kv: -kv[1])[:k_r]
            chosen = [j for j, _ in top]
            while len(chosen) < k_r:            # pad with self-repeats
                chosen.append(0)
            neigh[idx] = chosen
        self.neigh_items = torch.from_numpy(
            corpus_items[neigh.reshape(-1)].reshape(m, k_r, l_cap))
        if keep_corpus:
            self.corpus_items = torch.from_numpy(corpus_items)
        self.anchor_ids = torch.arange(m)
        if cache is not None:
            np.savez_compressed(Path(cache), items=self.neigh_items.numpy())

    def batch_relation(self, session_ids: torch.Tensor,
                       emb_table: torch.Tensor, control: str = "none",
                       detach_memory: bool = False
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (a_r, r0) for the given training-session indices.

        control -- revision M3, a *training-time* intervention on the
        retrieved memory (evaluation always uses control="none"):
          none    released top-K_r retrieval (default; bit-compatible with
                  every existing run record).
          rand    K_r uniformly random corpus sessions: destroys retrieval
                  relevance AND the memory-set distribution.
          shuffle the retrieved set of a random *other* session in the
                  batch: keeps the marginal distribution of memory sets and
                  breaks only the session<->memory correspondence.
        detach_memory -- control="frozen": stop gradient through the memory
        side (neighbor-item embeddings and relation adjacency) while the
        local path and scoring head stay trainable.
        """
        dev = emb_table.device
        if control == "rand":
            if self.corpus_items is None:
                raise ValueError("control='rand' requires keep_corpus=True "
                                 "at construction time")
            idx = torch.randint(0, self.n_corpus,
                                (len(session_ids), self.k_r))
            nb = self.corpus_items[idx].to(dev)                 # (B, K, L)
        elif control == "shuffle":
            perm = torch.randperm(len(session_ids))
            nb = self.neigh_items[session_ids[perm]].to(dev)    # (B, K, L)
        elif control == "none":
            nb = self.neigh_items[session_ids].to(dev)          # (B, K, L)
        else:
            raise ValueError(f"unknown rel-control {control!r}")
        mask = (nb > 0).float().unsqueeze(-1)
        r0 = (emb_table[nb] * mask).sum(2) / mask.sum(2).clamp_min(1.0)
        a = torch.nn.functional.cosine_similarity(
            r0.unsqueeze(2), r0.unsqueeze(1), dim=-1)
        eye = torch.eye(a.shape[-1], device=a.device).unsqueeze(0)
        a = a * (1 - eye)
        a = a / a.sum(-1, keepdim=True).clamp_min(1e-9)
        if detach_memory:
            a, r0 = a.detach(), r0.detach()
        return a, r0


def relation_matrix(retrieved: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build A_r (K_r x K_r) and the histogram features of retrieved sessions."""
    k = len(retrieved)
    n_items = 1 + max((max(s) for s in retrieved if s), default=0)
    hist = np.zeros((k, n_items), dtype=np.float32)
    for r, s in enumerate(retrieved):
        for i in s:
            hist[r, i] += 1.0
    norms = np.linalg.norm(hist, axis=1, keepdims=True)
    hist_n = hist / np.clip(norms, 1e-9, None)
    a_r = hist_n @ hist_n.T
    np.fill_diagonal(a_r, 0.0)  # avoid self-loops dominating the GCN
    a_r = a_r / np.clip(a_r.sum(axis=1, keepdims=True), 1e-9, None)
    return torch.tensor(a_r, dtype=torch.float32), torch.tensor(hist_n, dtype=torch.float32)
