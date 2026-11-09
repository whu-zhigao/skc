#!/usr/bin/env python3
"""Shared utilities for the ICDE experiment suite (A1-D3).

The suite auto-locates the repo code/ root (dir containing mplignn/);
results land in <root>/results/systems/. Every number that enters the
paper must be regenerable from those JSON files."""
from __future__ import annotations

import heapq
import json
import os
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def _find_code_root() -> Path:
    env = os.environ.get("SKC_CODE_ROOT")
    here = Path(__file__).resolve()
    cands = []
    if env:
        cands.append(Path(env))
    cands += [here.parents[1], here.parents[2] / "code",
              here.parents[3] / "code", Path.cwd(), Path.cwd() / "code"]
    # Any ancestor that DIRECTLY contains mplignn/ (e.g. the self-contained
    # cloud bundle layout <bundle>/code/{mplignn,data,experiments_icde}).
    cands += list(here.parents[0:4])
    for c in cands:
        if (c / "mplignn").is_dir():
            return c
    return here.parents[1]


CODE = _find_code_root()
sys.path.insert(0, str(CODE))
RESULTS = CODE / "results" / "systems"
REAL = CODE / "results" / "real"
# Dataset root: dir holding the <dataset>.pkl files. Default CODE/data;
# override with SKC_DATA_ROOT when only the data files are shipped to
# the node (GPU_GUIDE section (c)).
DATA_ROOT = (Path(os.environ["SKC_DATA_ROOT"]).resolve()
             if os.environ.get("SKC_DATA_ROOT") else CODE / "data")


def data_file(ds: str) -> Path:
    """Absolute path of a dataset pkl ('diginetica' or 'diginetica.pkl')."""
    name = ds if str(ds).endswith(".pkl") else f"{ds}.pkl"
    return DATA_ROOT / name


def ensure_results() -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    return RESULTS


def write_json(name, obj: dict) -> Path:
    """Write obj as JSON under results/systems/. `name` may be a bare
    filename (legacy top-level layout), a 'sub/dir/name.json' per-config
    path, or an absolute path (used by --out single-file mode)."""
    ensure_results()
    p = Path(name) if Path(name).is_absolute() else RESULTS / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))
    print(f"wrote {p}")
    return p


# -------------------------------------------- per-config outputs / resume
def config_out(exp: str, tag: str, out=None, force: bool = False):
    """Resolve the output path for ONE experiment configuration.

    Default layout: results/systems/<exp>/<tag>.json, where <tag>
    encodes dataset + key params + seed (e.g. diginetica_nq3000_k10).
    With --out (a path), legacy single-file mode: the given path is
    always (re)written and never skipped. Returns (path, skip): skip is
    True when the per-config file already exists and --force was not
    given (idempotent resume)."""
    if out:
        return Path(out), False
    p = RESULTS / exp / f"{tag}.json"
    return p, (p.exists() and not force)


def load_data(pkl: Path):
    from mplignn.data import load_sessions
    return load_sessions(pkl)


def serving_prefixes(sessions):
    """Return leakage-free next-item query prefixes.

    Evaluation sessions end in the held-out target item.  Retrieval must use
    the prefix ``s[:-1]``; passing the complete session would leak the target
    into Jaccard/MinHash neighbor search.
    """
    return [s[:-1] if len(s) > 1 else list(s) for s in sessions]


def load_model(data, ckpt=None, device="cpu"):
    import torch
    from mplignn.model import MPLI_GNN
    if ckpt is None:
        ckpt = REAL / "abl_no_residual_diginetica.pt"
    model = MPLI_GNN(n_items=data.n_items, d=64, local_layers=2,
                     n_heads=2, global_blocks=2, vocab_blocks=1, k_r=10,
                     no_residual=True)
    model.load_state_dict(
        torch.load(ckpt, map_location="cpu", weights_only=True))
    model.to(device).eval()
    return model


def _norm_neighbors(model, r0):
    """Enforce the model contract r0 == (B, k_r, d). Accepts a 2-D
    (K, d) pool as (1, K, d), truncates or repeat-pads the neighbor
    axis to the model's k_r, and HARD-fails on an embedding-dim
    mismatch (that is an upstream construction bug; never project it
    away silently)."""
    import torch
    if r0.dim() == 2:                              # (K,d) -> (1,K,d)
        r0 = r0.unsqueeze(0)
    k = r0.shape[1]
    want_d = getattr(model.embedding, "embedding_dim", r0.shape[2])
    if r0.shape[2] != want_d:
        raise ValueError(
            f"neighbor profile dim {r0.shape[2]} != model dim {want_d}; "
            "fix the profile construction upstream")
    k_r = getattr(getattr(model, "tare", None), "k_r", k)
    if k > k_r:
        r0 = r0[:, :k_r, :]
    elif k < k_r:
        reps = (k_r + k - 1) // k
        r0 = r0.repeat(1, reps, 1)[:, :k_r, :]
    return r0.float().contiguous()


def run_forward(model, item_seqs, neighbor_profiles):
    """SINGLE ADAPTATION POINT for MPLI_GNN.forward, wired to the real
    signature in mplignn/model.py:

        forward(seqs (B,T), lens (B,), a (B,T,T), a_r=None, r0=None)

    where a = build_adjacency(seqs, lens) (mplignn.data) and (a_r, r0)
    are the session-relation inputs of RelationEnhancement: a_r is the
    (B,K,K) relation adjacency and r0 the (B,K,d) initial relation
    representations. The suite's pooled neighbor-session profiles
    (B,K,d) play the role of r0; a_r = identity, so relation-graph
    smoothing degenerates to its skip term (r = r0 + r0/2) and r_bar is
    the pooled neighbor profile -- exactly the serving semantic SKC
    caches. Returns the full-sort scores tensor (B, n_items).
    Gradients flow when model.training (b7 --train co-training)."""
    import torch
    from mplignn.data import build_adjacency
    seqs = item_seqs
    lens = (seqs != 0).sum(dim=1).clamp(min=1).long()
    a = build_adjacency(seqs, lens)
    if neighbor_profiles is None:
        a_r = r0 = None
    else:
        r0 = _norm_neighbors(model, neighbor_profiles)
        k = r0.shape[1]
        a_r = (torch.eye(k, device=r0.device, dtype=torch.float32)
               .unsqueeze(0).expand(seqs.shape[0], k, k).contiguous())
    grad_enabled = bool(model.training) or (
        r0 is not None and r0.requires_grad
    )
    with torch.set_grad_enabled(grad_enabled):
        out = model(seqs, lens, a, a_r, r0)
    if isinstance(out, dict):
        return out["scores"]
    return out[0] if isinstance(out, (tuple, list)) else out


def pool_profiles(emb: np.ndarray, sessions) -> np.ndarray:
    out = np.zeros((len(sessions), emb.shape[1]), dtype=np.float32)
    for i, s in enumerate(sessions):
        out[i] = emb[np.asarray(sorted(set(s)))].mean(axis=0)
    return out


# ---------------------------------------------------------------- neighbors
def neighbor_ids(queries, corpus, k: int = 10):
    """Reuse the repo implementation; fall back to local exact Jaccard."""
    try:
        from systems.bench_retrieval import baseline_neighbor_ids
        return baseline_neighbor_ids(queries, corpus, k)
    except Exception:
        return jaccard_topk(queries, corpus, k)


def jaccard_topk(queries, corpus, k: int = 10) -> List[List[int]]:
    post: Dict[int, List[int]] = {}
    sizes = np.empty(len(corpus), dtype=np.int32)
    for sid, s in enumerate(corpus):
        sizes[sid] = len(set(s))
        for it in set(s):
            post.setdefault(it, []).append(sid)
    out = []
    for q in queries:
        qs = set(q)
        cnt: Dict[int, int] = {}
        for it in qs:
            for sid in post.get(it, ()):
                cnt[sid] = cnt.get(sid, 0) + 1
        scored = ((sid, c / (len(qs) + sizes[sid] - c))
                  for sid, c in cnt.items())
        out.append([sid for sid, _ in heapq.nlargest(k, scored,
                                                     key=lambda x: x[1])])
    return out


# ------------------------------------------------------------------ metrics
def gini(freq) -> float:
    x = np.sort(np.asarray(list(freq), dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def ranking_metrics(logits: np.ndarray, targets: np.ndarray,
                    ks=(20,)) -> Dict[str, float]:
    order = np.argsort(-logits, axis=1)
    ranks = np.array([np.where(order[i] == targets[i])[0][0] + 1
                      for i in range(len(targets))])
    out = {f"HR@{k}": float((ranks <= k).mean()) for k in ks}
    out["MRR@10"] = float((1.0 / np.maximum(ranks, 1))[ranks <= 10].sum()
                          / len(ranks))
    return out


# ------------------------------------------------------------- timing utils
def timeit(fn, warmup: int = 10, iters: int = 100):
    import torch
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    a = np.array(ts)
    return {"p50": float(np.percentile(a, 50)),
            "p99": float(np.percentile(a, 99)), "mean": float(a.mean())}


# ------------------------------------------------------------ cache plugins
class CountMin:
    """Vectorized Count-Min sketch (Cormode & Muthukrishnan)."""

    def __init__(self, w: int = 4096, d: int = 4, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w, self.d = w, d
        self.a = rng.integers(1, 1 << 31, size=d, dtype=np.int64)
        self.b = rng.integers(0, 1 << 31, size=d, dtype=np.int64)
        self.table = np.zeros((d, w), dtype=np.int64)
        self._p = (1 << 61) - 1

    def _idx(self, ids: np.ndarray) -> np.ndarray:  # (len(ids), d)
        return ((ids[:, None] * self.a[None, :]
                 + self.b[None, :]) % self._p) % self.w

    def update(self, ids: Sequence[int]):
        idx = self._idx(np.asarray(ids, dtype=np.int64))
        for r in range(self.d):
            np.add.at(self.table[r], idx[:, r], 1)

    def estimate(self, ids: np.ndarray) -> np.ndarray:
        idx = self._idx(ids.astype(np.int64))
        return self.table[np.arange(self.d)[None, :], idx].min(axis=1)

    def bytes(self) -> int:
        return self.table.nbytes


class ARC:
    """ARC replacement cache (Megiddo & Modha, FAST'03), standard
    simplified port. request(x) -> True on hit."""

    def __init__(self, capacity: int):
        self.c = max(1, capacity)
        self.p = self.c / 2.0
        self.t1: OrderedDict = OrderedDict()
        self.t2: OrderedDict = OrderedDict()
        self.b1: OrderedDict = OrderedDict()
        self.b2: OrderedDict = OrderedDict()

    def _replace(self):
        if self.t1 and len(self.t1) > self.p:
            lru, _ = self.t1.popitem(last=False)
            self.b1[lru] = None
        elif self.t2:
            lru, _ = self.t2.popitem(last=False)
            self.b2[lru] = None

    def request(self, x) -> bool:
        if x in self.t1:
            del self.t1[x]; self.t2[x] = None; return True
        if x in self.t2:
            self.t2.move_to_end(x); return True
        if x in self.b1:
            self.p = min(self.c, self.p + max(1, len(self.b2)
                                              // max(1, len(self.b1))))
            self._replace()
            del self.b1[x]; self.t2[x] = None; return False
        if x in self.b2:
            self.p = max(0.0, self.p - max(1, len(self.b1)
                                           // max(1, len(self.b2))))
            self._replace()
            del self.b2[x]; self.t2[x] = None; return False
        if len(self.t1) + len(self.t2) >= self.c:
            if len(self.t1) + len(self.b1) <= self.c:
                if self.b2:
                    self.b2.popitem(last=False)
                self._replace()
            elif self.t1:
                lru, _ = self.t1.popitem(last=False)
                self.b1[lru] = None
        elif len(self.b1) + len(self.b2) >= self.c:
            if len(self.b1) + len(self.b2) == 2 * self.c and self.b1:
                self.b1.popitem(last=False)
            self._replace()
        self.t1[x] = None
        return False


class LRUCache:
    def __init__(self, capacity: int):
        self.c = max(1, capacity)
        self.d: OrderedDict = OrderedDict()

    def request(self, x) -> bool:
        if x in self.d:
            self.d.move_to_end(x); return True
        if len(self.d) >= self.c:
            self.d.popitem(last=False)
        self.d[x] = None
        return False


def _cm_estimate1(cm, x) -> int:
    return int(cm.estimate(np.asarray([x], dtype=np.int64))[0])


class TinyLFU:
    """TinyLFu admission policy (Einziger, Friedman & Manes, "TinyLFU:
    A Highly Efficient Cache Admission Policy", USENIX SCN 2017): a new
    entry is admitted only if its TinyLFU frequency estimate (doorkeeper
    + Count-Min sketch, halved every `reset_window` requests to age
    history) EXCEEDS that of the LRU victim in the cache. Composed with
    an LRU victim order here (the paper composes it with SLRU -- see
    WTinyLFU below). request(x) -> True on hit."""

    def __init__(self, capacity: int, count_w: int = 4096,
                 count_d: int = 2, seed: int = 0):
        self.c = max(1, capacity)
        self.cm = CountMin(w=count_w, d=count_d, seed=seed)
        self.door: OrderedDict = OrderedDict()   # FIFO doorkeeper
        self.door_cap = self.c
        self.cache: OrderedDict = OrderedDict()  # LRU order; victim=first
        self._seen = 0
        self.reset_window = max(1024, 4 * self.c)

    def _tick(self):
        self._seen += 1
        if self._seen % self.reset_window == 0:
            self.cm.table //= 2                  # periodic halving (Reset)
            self.door.clear()

    def _freq(self, x) -> int:
        return _cm_estimate1(self.cm, x) + (1 if x in self.door else 0)

    def request(self, x) -> bool:
        self._tick()
        if x in self.door:
            self.cm.update((x,))                 # counted from 2nd sight
        else:
            self.door[x] = None
            if len(self.door) > self.door_cap:
                self.door.popitem(last=False)
        if x in self.cache:
            self.cache.move_to_end(x)
            return True
        if len(self.cache) < self.c:
            self.cache[x] = None
        else:
            victim = next(iter(self.cache))
            if self._freq(x) > self._freq(victim):
                del self.cache[victim]
                self.cache[x] = None
        return False


class WTinyLFU:
    """W-TinyLFU (Einziger, Roy, Bain & Lipshtes, "Adaptive Software
    Cache Management with AdaptiveTinyLFU", ACM TOS 2018): a small
    windowed-LRU (1% of capacity) admits new entries unconditionally;
    only the window tail competes for the SLRU main space (t1 20% /
    t2 80%), where TinyLFU admission compares it against the LRU victim
    (ghost lists g1/g2 bias the victim choice). Frequency from the same
    doorkeeper + Count-Min sketch with periodic halving.
    request(x) -> True on hit (window, t1 or t2). Standard simplified
    port of the TOS'18 structure."""

    def __init__(self, capacity: int, count_w: int = 4096,
                 count_d: int = 2, seed: int = 0):
        self.c = max(2, capacity)
        self.win_cap = max(1, self.c // 100)
        main = max(1, self.c - self.win_cap)
        self.t1_cap = max(1, int(0.2 * main))
        self.t2_cap = max(1, main - self.t1_cap)
        self.window: OrderedDict = OrderedDict()
        self.t1: OrderedDict = OrderedDict()
        self.t2: OrderedDict = OrderedDict()
        self.g1: OrderedDict = OrderedDict()
        self.g2: OrderedDict = OrderedDict()
        self.cm = CountMin(w=count_w, d=count_d, seed=seed)
        self.door: OrderedDict = OrderedDict()
        self.door_cap = self.c
        self._seen = 0
        self.reset_window = max(1024, 4 * self.c)

    def _tick(self):
        self._seen += 1
        if self._seen % self.reset_window == 0:
            self.cm.table //= 2
            self.door.clear()

    def _freq(self, x) -> int:
        return _cm_estimate1(self.cm, x) + (1 if x in self.door else 0)

    def _push_ghost(self, d: OrderedDict, x):
        d[x] = None
        cap = self.t1_cap if d is self.g1 else self.t2_cap
        while len(d) > cap:
            d.popitem(last=False)

    def _admit_main(self, cand):
        # SLRU rebalance: promote t1 LRU into t2 when t1 is over cap.
        if len(self.t1) > self.t1_cap:
            ev, _ = self.t1.popitem(last=False)
            self.t2[ev] = None
        if len(self.t2) > self.t2_cap:
            ev, _ = self.t2.popitem(last=False)
            self._push_ghost(self.g2, ev)
        if len(self.t1) < self.t1_cap:
            self.t1[cand] = None
            return
        # both segments full: TinyLFU compare against a ghost-biased
        # victim (candidate seen in a ghost list -> t2 victim).
        victim_d = (self.t2
                    if (cand in self.g1 or cand in self.g2) else self.t1)
        if not victim_d:
            self.t1[cand] = None
            return
        victim, _ = victim_d.popitem(last=False)
        if self._freq(cand) > self._freq(victim):
            self.t1[cand] = None
            self._push_ghost(self.g1 if victim_d is self.t1 else self.g2,
                             victim)
        else:                                     # rejected: restore victim
            victim_d[victim] = None
            victim_d.move_to_end(victim, last=False)

    def request(self, x) -> bool:
        self._tick()
        if x in self.door:
            self.cm.update((x,))
        else:
            self.door[x] = None
            if len(self.door) > self.door_cap:
                self.door.popitem(last=False)
        if x in self.window:
            self.window.move_to_end(x); return True
        if x in self.t1:
            del self.t1[x]; self.t2[x] = None
            if len(self.t2) > self.t2_cap:
                ev, _ = self.t2.popitem(last=False)
                self._push_ghost(self.g2, ev)
            return True
        if x in self.t2:
            self.t2.move_to_end(x); return True
        # miss: new reference enters the window; the window tail then
        # competes for the main space (W-TinyLFU admission).
        self.window[x] = None
        while len(self.window) > self.win_cap:
            cand, _ = self.window.popitem(last=False)
            self._admit_main(cand)
        return False


class LeCaR:
    """LeCaR (Shi, Yu, Tang & Kaashoek, "LeCaR: Latency-Critical
    Adaptive Replacement Cache", HotNets 2018): two LRU lists --
    recency (l1) and frequency (l2) -- whose eviction weights w1/w2 are
    learned online (exponential updates, learning rate eta) from hits
    and ghost hits; evicted blocks enter ghost lists that credit the
    OTHER expert on later re-references with reward r = 1/(1+distance),
    distance = number of references since eviction. New references
    enter l1. request(x) -> True on hit."""

    def __init__(self, capacity: int, eta: float = 0.45, seed: int = 0):
        self.c = max(2, capacity)
        self.l1: OrderedDict = OrderedDict()
        self.l2: OrderedDict = OrderedDict()
        self.g1: OrderedDict = OrderedDict()      # ghost: sid -> eviction t
        self.g2: OrderedDict = OrderedDict()
        self.w1 = 0.5
        self.eta = eta
        self.rng = np.random.default_rng(seed)
        self._t = 0

    def _credit(self, expert: int, r: float):
        """Exponential update toward the credited expert."""
        if expert == 1:
            self.w1 = self.w1 + self.eta * r * (1.0 - self.w1)
        else:
            self.w1 = self.w1 - self.eta * r * self.w1
        self.w1 = float(np.clip(self.w1, 0.0, 1.0))

    def _push_ghost(self, d: OrderedDict, x):
        d[x] = self._t
        while len(d) > self.c:
            d.popitem(last=False)

    def request(self, x) -> bool:
        self._t += 1
        if x in self.l1:
            self.l1.move_to_end(x); self._credit(1, 1.0); return True
        if x in self.l2:
            self.l2.move_to_end(x); self._credit(2, 1.0); return True
        reward = None                            # ghost hit -> other expert
        if x in self.g1:
            reward = (2, 1.0 / (1.0 + (self._t - self.g1[x])))
        elif x in self.g2:
            reward = (1, 1.0 / (1.0 + (self._t - self.g2[x])))
        if len(self.l1) + len(self.l2) >= self.c:
            take_l1 = (self.l1 and (not self.l2
                                    or self.rng.random() < self.w1))
            if take_l1:
                ev, _ = self.l1.popitem(last=False)
                self._push_ghost(self.g1, ev)
            else:
                ev, _ = self.l2.popitem(last=False)
                self._push_ghost(self.g2, ev)
        self.l1[x] = None
        if reward:
            self._credit(*reward)
        return False


class GreedyDualSize:
    """GreedyDual-Size (Cao & Irani, "Cost-Aware WWW Proxy Caching
    Algorithms", SOSP 1997): every cached object carries a value
    H(x) = L + cost(x)/size(x), where L is the H of the most recent
    eviction; the smallest H is evicted and a hit refreshes
    H(x) <- max(H(x), L). This port uses the frequency-cost variant
    (cost(x) = 1 + Count-Min frequency estimate; the GD-Frequency
    direction Cao & Irani discuss) with optional per-object `sizes`,
    since Tier-1 profiles have a uniform byte footprint (uniform
    sizes/costs degenerate GreedyDual-Size to plain LRU).
    request(x) -> True on hit."""

    def __init__(self, capacity: int, sizes=None, count_w: int = 4096,
                 count_d: int = 2, seed: int = 0):
        self.c = max(1, capacity)
        self.sizes = dict(sizes) if sizes else {}
        self.cm = CountMin(w=count_w, d=count_d, seed=seed)
        self.h: Dict = {}          # key -> (H, tie) of the live entry
        self.heap: List = []       # (H, tie, key); stale ties skipped
        self.tie = 0
        self.L = 0.0

    def request(self, x) -> bool:
        self.cm.update((x,))
        cost = 1.0 + _cm_estimate1(self.cm, x)
        size = float(self.sizes.get(x, 1.0))
        cur = self.h.get(x)
        if cur is not None:                        # hit: refresh H
            H = max(cur[0], self.L)
            self.tie += 1
            heapq.heappush(self.heap, (H, self.tie, x))
            self.h[x] = (H, self.tie)
            return True
        while len(self.h) >= self.c:               # evict smallest live H
            H, tie, key = heapq.heappop(self.heap)
            live = self.h.get(key)
            if live is None or live[1] != tie:     # stale entry
                continue
            del self.h[key]
            self.L = max(self.L, H)
            break
        self.tie += 1
        heapq.heappush(self.heap, (self.L + cost / size, self.tie, x))
        self.h[x] = (self.L + cost / size, self.tie)
        return False


# Online admission caches that expose the request(x)->bool stream
# interface (LFU stays special-cased in exp_b1: periodic re-admission).
ONLINE_POLICIES = ("lru", "arc", "tinylfu", "wtinylfu", "lecar",
                   "greedydualsize")
ADMISSION_CITATIONS = {
    "lru": "classic LRU (baseline)",
    "lfu": "perfect-LFU with periodic re-admission (baseline)",
    "arc": "Megiddo & Modha, FAST 2003",
    "tinylfu": "Einziger, Friedman & Manes, SCN 2017",
    "wtinylfu": "Einziger, Roy, Bain & Lipshtes, ACM TOS 2018",
    "lecar": "Shi, Yu, Tang & Kaashoek, HotNets 2018",
    "greedydualsize": "Cao & Irani, SOSP 1997",
}


def make_admission_cache(policy: str, capacity: int, seed: int = 0):
    """Factory for the online admission/replacement caches."""
    if policy == "lru":
        return LRUCache(capacity)
    if policy == "arc":
        return ARC(capacity)
    if policy == "tinylfu":
        return TinyLFU(capacity, seed=seed)
    if policy == "wtinylfu":
        return WTinyLFU(capacity, seed=seed)
    if policy == "lecar":
        return LeCaR(capacity, seed=seed)
    if policy == "greedydualsize":
        return GreedyDualSize(capacity, seed=seed)
    raise ValueError(f"unknown admission policy {policy!r} "
                     f"(valid: {ONLINE_POLICIES})")


def simulate_admitted(policy: str, capacity: int, nbrs, seed: int = 0):
    """Replay the neighbor-id stream through an online admission cache
    (arrival order) and return the FINAL resident set as a sorted numpy
    array -- the admission set an offline builder (B5) would materialize
    under that policy."""
    cache = make_admission_cache(policy, capacity, seed=seed)
    for nb in nbrs:
        for sid in nb:
            cache.request(sid)
    if isinstance(cache, LRUCache):
        keys = list(cache.d.keys())
    elif isinstance(cache, ARC):
        keys = list(cache.t1.keys()) + list(cache.t2.keys())
    elif isinstance(cache, TinyLFU):
        keys = list(cache.cache.keys())
    elif isinstance(cache, WTinyLFU):
        keys = (list(cache.window.keys()) + list(cache.t1.keys())
                + list(cache.t2.keys()))
    elif isinstance(cache, LeCaR):
        keys = list(cache.l1.keys()) + list(cache.l2.keys())
    else:  # GreedyDualSize
        keys = list(cache.h.keys())
    return np.sort(np.asarray(keys, dtype=np.int64))


# ------------------------------------------- quantization / admission (B5-B7)
def quantize_rows(X: np.ndarray, bits: int):
    """Symmetric per-row quantization for bits in {8,4,2}.

    qmax = 2**(bits-1) - 1; per-row scale s = max|row| / qmax (float32,
    one per row); codes = clip(round(X / s), -qmax, qmax) stored int8
    for every width (nominal widths below 8 bit are accounted
    analytically by quantized_rows_bytes / mixed_rows_bytes)."""
    if bits not in (8, 4, 2):
        raise ValueError(f"bits must be in {{8, 4, 2}}, got {bits}")
    X = np.asarray(X, dtype=np.float32)
    qmax = float(2 ** (bits - 1) - 1)
    scales = np.maximum(np.abs(X).max(axis=1, keepdims=True), 1e-12) / qmax
    codes = np.clip(np.rint(X / scales), -qmax, qmax).astype(np.int8)
    return codes, scales.astype(np.float32)


def dequantize_rows(codes: np.ndarray, scales: np.ndarray, bits: int):
    """Inverse of quantize_rows (bits kept for interface symmetry; the
    level information lives in the codes themselves)."""
    _ = bits
    return (np.asarray(codes, dtype=np.float32)
            * np.asarray(scales, dtype=np.float32)).astype(np.float32)


def _row_bytes(d: int, bits: int) -> float:
    return float(np.ceil(d * bits / 8.0)) + 4.0  # codes + float32 scale


def quantized_rows_bytes(n_rows: int, d: int, bits: int) -> int:
    """Analytic bytes for n_rows uniform-precision quantized profiles."""
    return int(n_rows * _row_bytes(d, bits))


def mixed_rows_bytes(bits: np.ndarray, d: int) -> int:
    """Analytic bytes for per-row mixed-precision profiles."""
    b = np.asarray(bits)
    return int(np.sum(np.ceil(d * b.astype(np.float64) / 8.0) + 4.0))


def mixed_precision_partition(utilities, fracs=(0.25, 0.50, 0.25)):
    """Assign per-row bit-widths (8/4/2) by DESCENDING utility:
    head fracs[0] -> int8, mid fracs[1] -> int4, tail fracs[2] -> int2.
    Cuts are rounded on the sorted order so every row is assigned."""
    u = np.asarray(utilities, dtype=np.float64).ravel()
    n = u.size
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    if abs(sum(float(f) for f in fracs) - 1.0) > 1e-9:
        raise ValueError("fracs must sum to 1")
    order = np.argsort(-u, kind="stable")
    cut1 = int(round(float(fracs[0]) * n))
    cut2 = min(n, cut1 + int(round(float(fracs[1]) * n)))
    cut1 = min(cut1, cut2)
    bits = np.empty(n, dtype=np.int8)
    bits[order[:cut1]] = 8
    bits[order[cut1:cut2]] = 4
    bits[order[cut2:]] = 2
    return bits


def _knee_index(curve: np.ndarray) -> int:
    """kneedle-lite: index of max distance-to-chord on a descending
    curve (utilities sorted descending, normalized so both endpoints of
    the chord are (0,0)/(1,1); flat curves split in the middle)."""
    n = curve.size
    if n < 3:
        return max(1, n - 1)
    y = curve.astype(np.float64)
    if y[0] == y[-1]:
        return n // 2
    yn = (y - y[0]) / (y[-1] - y[0])
    x = np.arange(n, dtype=np.float64) / (n - 1)
    return max(1, int(np.argmax(np.abs(yn - x))))


def elbow_mixed_partition(utilities):
    """Data-driven 8/4/2 split: knee #1 on the sorted utility curve
    bounds the int8 head; knee #2 on the remaining tail bounds int4 vs
    int2 (both knees via max distance-to-chord). Falls back to all-int8
    when fewer than 3 rows."""
    u = np.asarray(utilities, dtype=np.float64).ravel()
    n = u.size
    if n < 3:
        return np.full(n, 8, dtype=np.int8)
    order = np.argsort(-u, kind="stable")
    us = u[order]
    k1 = int(np.clip(_knee_index(us), 1, n - 2))
    tail = us[k1:]
    k2 = k1 + int(np.clip(_knee_index(tail), 1, tail.size))
    k2 = int(np.clip(k2, k1, n))
    bits = np.full(n, 2, dtype=np.int8)
    bits[order[:k1]] = 8
    bits[order[k1:k2]] = 4
    bits[order[k2:]] = 2
    return bits


def cars_admission_score(freq, gain, cost, staleness):
    """CARS admission score = freq * (gain - cost) - staleness.
    Per-candidate quantities broadcast as numpy arrays (B5 passes
    staleness=0; B7 supplies real drift norms)."""
    return (np.asarray(freq, dtype=np.float64)
            * (np.asarray(gain, dtype=np.float64)
               - np.asarray(cost, dtype=np.float64))
            - np.asarray(staleness, dtype=np.float64))
