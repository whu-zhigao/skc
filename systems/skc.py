"""Session KV cache with graph-aware admission and safe page maintenance.

The class is intentionally small and dependency-light: systems benchmarks can
use it as a reference implementation without importing a training framework.
It implements the semantics used by the paper:

* pages are materialized as units, not as isolated rows;
* misses fall back to a caller-supplied immutable source matrix;
* NFA eviction scores a page by the aggregate frequency of its sessions;
* replacement pages are copied and atomically published.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class Page:
    ids: Tuple[int, ...]
    profiles: np.ndarray
    access_count: int = 0
    last_access: int = 0
    generation: int = 0
    nbytes: int = 0

    def __post_init__(self) -> None:
        if self.profiles.ndim != 2 or len(self.ids) != len(self.profiles):
            raise ValueError("page ids and profiles must have equal length")
        # Page profiles are private copies.  Once published they are immutable;
        # a refresh creates and publishes a new Page object.
        self.profiles = np.ascontiguousarray(self.profiles,
                                             dtype=np.float32)
        self.profiles.flags.writeable = False
        self.nbytes = self.profiles.nbytes


@dataclass
class SKCConfig:
    page_size: int = 128
    memory_budget_mb: float = 50.0
    d: int = 64
    admission: str = "nfa"
    invalidation_theta: float = 0.01
    quant_hot: str = "fp32"
    quant_warm: str = "int8"


class SessionKVCache:
    """Paged profile cache with neighbor-aware admission."""

    def __init__(self, cfg: SKCConfig,
                 source_profiles: Optional[np.ndarray] = None,
                 n_sessions: Optional[int] = None):
        self.cfg = cfg
        if source_profiles is None and n_sessions is None:
            raise ValueError("provide source_profiles or n_sessions")
        self.source = (None if source_profiles is None
                       else np.asarray(source_profiles, dtype=np.float32))
        self.n_sessions = (len(source_profiles) if source_profiles is not None
                           else int(n_sessions))
        if self.source is not None and len(self.source) != self.n_sessions:
            raise ValueError("source_profiles length does not match n_sessions")
        if self.source is not None and self.source.shape[1] != cfg.d:
            raise ValueError("source profile dimension does not match cfg.d")

        self.pages: Dict[int, Page] = {}
        self.neighbor_freq: Counter = Counter()
        self.tick = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.max_bytes = int(cfg.memory_budget_mb * 1e6)
        self._lock = threading.RLock()

    # ------------------------------------------------------------ admission
    def _page_range(self, pid: int) -> Tuple[int, int]:
        start = pid * self.cfg.page_size
        return start, min(start + self.cfg.page_size, self.n_sessions)

    def _page_id(self, sid: int) -> int:
        return int(sid) // self.cfg.page_size

    def _materialize_page(self, pid: int,
                          profiles: Optional[np.ndarray] = None) -> Page:
        start, end = self._page_range(pid)
        if start >= end:
            raise KeyError(f"empty page {pid}")
        matrix = self.source if profiles is None else profiles
        if matrix is None:
            raise ValueError(f"no source profiles for page {pid}")
        ids = tuple(range(start, end))
        return Page(ids=ids, profiles=matrix[list(ids)].copy())

    def observe_neighbors(self, neighbor_ids: Sequence[int]) -> None:
        """Record which sessions served as neighbors."""
        with self._lock:
            self.neighbor_freq.update(int(x) for x in neighbor_ids)

    def admit(self, session_ids: Sequence[int],
              profiles: Optional[np.ndarray] = None,
              generation: int = 0) -> None:
        """Materialize complete pages containing the requested sessions.

        Admitting complete pages is important for consistency maintenance:
        page-local ledgers and refresh decisions cannot be defined over
        partially populated rows.
        """
        session_ids = [int(x) for x in session_ids]
        with self._lock:
            self.tick += 1
            for sid in session_ids:
                if not 0 <= sid < self.n_sessions:
                    raise IndexError(f"session {sid} outside cache")
                pid = self._page_id(sid)
                if pid not in self.pages:
                    self.pages[pid] = self._materialize_page(pid, profiles)
                page = self.pages[pid]
                page.access_count += 1
                page.last_access = self.tick
                if page.generation != generation:
                    page.generation = generation
            self._evict_if_needed_locked()

    def _page_frequency(self, page: Page) -> float:
        """Aggregate NFA value of a page, not the accidental page-id key."""
        return float(sum(self.neighbor_freq.get(sid, 0)
                         for sid in page.ids))

    def _evict_if_needed_locked(self) -> None:
        total = sum(page.nbytes for page in self.pages.values())
        while total > self.max_bytes and self.pages:
            if self.cfg.admission == "nfa":
                victim = min(self.pages,
                             key=lambda pid: self._page_frequency(self.pages[pid]))
            elif self.cfg.admission == "lru":
                victim = min(self.pages,
                             key=lambda pid: self.pages[pid].last_access)
            else:
                raise ValueError(f"unknown admission policy {self.admission}")
            total -= self.pages[victim].nbytes
            del self.pages[victim]
            self.evictions += 1

    def _evict_if_needed(self) -> None:
        with self._lock:
            self._evict_if_needed_locked()

    # ---------------------------------------------------------------- reads
    def fetch_profiles(self, session_ids: np.ndarray,
                       source_profiles: Optional[np.ndarray] = None
                       ) -> Tuple[np.ndarray, float]:
        """Serve profiles and recompute true misses from the source matrix.

        Misses are never represented by zero vectors.  If no source matrix is
        available, the method raises rather than silently returning invalid
        model inputs.
        """
        ids = np.asarray(session_ids, dtype=np.int64).tolist()
        if not ids:
            return np.empty((0, self.cfg.d), dtype=np.float32), 0.0

        source = self.source if source_profiles is None else source_profiles
        if source is None:
            raise ValueError("fetch_profiles requires source_profiles")
        source = np.asarray(source, dtype=np.float32)

        # Snapshot immutable pages under the short lock, then gather outside it.
        with self._lock:
            self.tick += 1
            snapshots = {pid: (page, page.profiles)
                         for pid, page in self.pages.items()}
            present_pages = set(snapshots)

        by_id = {sid: (snapshots[pid][0], row, snapshots[pid][1][row])
                 for pid, (page, matrix) in snapshots.items()
                 for row, sid in enumerate(page.ids)}
        result = np.empty((len(ids), self.cfg.d), dtype=np.float32)
        hit_mask = np.zeros(len(ids), dtype=bool)

        for i, sid in enumerate(ids):
            if sid in by_id:
                _, _, row_values = by_id[sid]
                result[i] = row_values
                hit_mask[i] = True
            else:
                result[i] = source[sid]

        with self._lock:
            for i, sid in enumerate(ids):
                if not hit_mask[i]:
                    self.misses += 1
                else:
                    self.hits += 1
            self.tick += 1
        hit_rate = float(hit_mask.mean())

        # Materialize any page that produced a miss, after producing a correct
        # response.  This keeps latency semantics explicit: service first,
        # then maintain.
        missing_pages = sorted({self._page_id(sid)
                                for sid, hit in zip(ids, hit_mask)
                                if not hit and self._page_id(sid) not in present_pages})
        if missing_pages:
            self.admit([sid for sid in ids if self._page_id(sid) in
                        set(missing_pages)], source, generation=0)
        return result, hit_rate

    # ---------------------------------------------------------------- DAI
    def invalidate_delta(self, emb_old: torch.Tensor,
                         emb_new: torch.Tensor,
                         item_to_sessions: Dict[int, List[int]]) -> int:
        """Invalidate pages whose source item drift exceeded the threshold."""
        delta = (emb_new.cpu() - emb_old.cpu()).norm(dim=-1)
        moved_items = set((delta > self.cfg.invalidation_theta)
                          .nonzero(as_tuple=False).flatten().tolist())
        invalid_pages = set()
        for item in moved_items:
            for sid in item_to_sessions.get(int(item), []):
                invalid_pages.add(self._page_id(int(sid)))
        with self._lock:
            for pid in invalid_pages:
                self.pages.pop(pid, None)
        return len(invalid_pages)

    def stats(self) -> Dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / max(1, total),
                "pages": len(self.pages),
                "evictions": self.evictions,
                "bytes": sum(page.nbytes for page in self.pages.values()),
                "budget_mb": self.cfg.memory_budget_mb,
            }


class OracleCache:
    """Offline budgeted coverage used only as a replay ceiling."""

    def __init__(self, budget_sessions: int):
        self.budget = budget_sessions
        self.cached: set[int] = set()

    def plan(self, neighbor_freq: Counter) -> None:
        top = [sid for sid, _ in neighbor_freq.most_common(self.budget)]
        self.cached = set(top)

    def hit_rate(self, queries: List[List[int]]) -> float:
        total, hit = 0, 0
        for nb_list in queries:
            for sid in nb_list:
                total += 1
                hit += int(sid in self.cached)
        return hit / max(1, total)
