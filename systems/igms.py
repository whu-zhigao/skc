"""IGMS: Incremental Global-Memory Store for session-based recommendation.

This module replaces the static, rebuild-per-corpus relation index of the
reference implementation (FastRelationIndex) with a data-store view of the
cross-session memory:

  key   = MinHash signature of the session's item set (b bands x r rows)
  value = pooled item-embedding profile, int8-quantized per dimension

Three properties the static index does not have:
  1. streaming inserts in O(len(session)) with no global rebuild;
  2. approximate top-K neighbor search through band-probing with a
     recall/latency knob (max_cand, bands);
  3. a value store whose footprint is 1 byte per dimension instead of
     K_r x L int64 item ids per anchor, assembled at batch time by a
     dequantization instead of an embedding-table gather + mean.

Ground-truth utilities (exact Jaccard top-K over the corpus) are included
so recall numbers can always be recomputed from raw data.
"""
from __future__ import annotations

import time
import sys
import threading
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
from pathlib import Path

import numpy as np

_P1, _P2 = (1 << 31) - 1, 4294967311  # hashing primes (31-bit workspace)


def _splitmix64(value: int) -> int:
    """Deterministic 64-bit mixer used for persisted posting keys."""
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) \
        & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) \
        & 0xFFFFFFFFFFFFFFFF
    return (value ^ (value >> 31)) & 0xFFFFFFFFFFFFFFFF


def _stable_band_key(chunk: bytes) -> int:
    """Hash a MinHash band independently of Python's process-random hash."""
    return _splitmix64(int.from_bytes(chunk, "little")) & 0xFFFFFF


def minhash_signature(items: Sequence[int], m: int, seed: int = 0,
                      ha=None, hb=None) -> np.ndarray:
    """m-permutation MinHash of the item set, uint32 vector. Permutation
    parameters may be supplied once by the index (hot path)."""
    if ha is None:
        rng = np.random.RandomState(seed)
        ha = rng.randint(1, _P1, size=m).astype(np.int64)
        hb = rng.randint(0, _P1, size=m).astype(np.int64)
    x = np.asarray(sorted(set(int(i) for i in items)), dtype=np.int64)
    return (((x[None, :] * ha[:, None] + hb[:, None]) % _P2)
            & 0xFFFFFFFF).min(axis=1).astype(np.uint32)


@dataclass
class IGMSConfig:
    sig_len: int = 64          # m = b * r
    bands: int = 32            # b (rows per band r = m / b)
    k_r: int = 10
    max_cand: int = 512        # candidate cap per query
    value_dim: int = 64
    quant: str = "int8"        # "int8" | "fp16" | "none"
    rank: str = "jaccard"      # "jaccard" (keeps item sets) | "profile"
    keep_item_sets: bool = True  # False = lean mode, sets dropped after
    compact_item_sets: bool = False  # freeze(): Python sets -> compact CSR
    discard_signatures: bool = False  # frozen blocks: postings subsume sigs
    bucket_layout: str = "dict"  # "dict" | "packed" | "sorted"
    seed: int = 0              #   (packed: one int64 array per band,
                              #   band<<56 | key<<32 | sid, vectorized probes)


class IGMS:
    """The incremental store. Insert-only by design (append-only logs are
    the natural production source); eviction hooks are provided."""

    def __init__(self, cfg: Optional[IGMSConfig] = None):
        self.cfg = cfg or IGMSConfig()
        assert self.cfg.sig_len % self.cfg.bands == 0
        self.rows = self.cfg.sig_len // self.cfg.bands
        rng = np.random.RandomState(self.cfg.seed)
        self._ha = rng.randint(1, _P1, size=self.cfg.sig_len).astype(np.int64)
        self._hb = rng.randint(0, _P1, size=self.cfg.sig_len).astype(np.int64)
        self.sigs: List[np.ndarray] = []            # per-session signatures
        self.item_sets: List[set] = []              # for exact re-ranking
        self.item_indptr: Optional[np.ndarray] = None
        self.item_ids: Optional[np.ndarray] = None
        self.buckets: Dict[Tuple[int, int, bytes], List[int]] = defaultdict(list) \
            if self.cfg.bucket_layout == "dict" else {}
        self.bands: Optional[List[np.ndarray]] = None   # latest frozen block
        self.posting_blocks: List[Optional[None]] = []
        self.posting_index: List[np.ndarray] = []
        self.block_first_sid: List[int] = []
        self.block_n_sessions: List[int] = []
        self._band_buf: Optional[List[List[int]]] = \
            [[] for _ in range(self.cfg.bands)] \
            if self.cfg.bucket_layout in ("packed", "sorted") else None
        self._dirty = False
        self.values: Optional[np.ndarray] = None    # (n, d) quantized store
        self._row_scale: Optional[np.ndarray] = None  # (n,) dequant scales
        self.n_insert = 0
        self.insert_seconds = 0.0
        self.probe_ops = 0                           # candidates examined
        self._lock = threading.RLock()
        self._deleted_sessions: set[int] = set()

    # ------------------------------------------------------------------ write
    def insert(self, items: Sequence[int],
               pooled: Optional[np.ndarray] = None) -> int:
        """Insert one session in O(len(items)+b) amortized time."""
        t0 = time.perf_counter()
        with self._lock:
            sid = self.n_insert
            if sid >= (1 << 32):
                raise OverflowError(
                    "compact postings support at most 2^32-1 sessions; "
                    "shard the corpus or widen the posting word")
            sig = minhash_signature(items, self.cfg.sig_len, self.cfg.seed,
                                    self._ha, self._hb)
            self.sigs.append(sig)
            session_items = set(int(i) for i in items)
            self.item_sets.append(
                session_items if self.cfg.keep_item_sets else set())
            sig_b = sig.tobytes()
            row = self.rows
            if self._band_buf is not None:
                for bix in range(self.cfg.bands):
                    chunk = sig_b[bix * 4 * row:(bix + 1) * 4 * row]
                    # Posting layout: band index (8 bits), stable key
                    # (24 bits), and session id (32 bits).  The layout is
                    # process-stable and lets one vectorized lookup search all
                    # bands in a frozen block.
                    self._band_buf[bix].append(
                        (bix << 56) |
                        (_stable_band_key(chunk) << 32) | sid)
                self._dirty = True
            else:
                for bix in range(self.cfg.bands):
                    key = (bix, row,
                           sig_b[bix * 4 * row:(bix + 1) * 4 * row])
                    self.buckets[key].append(sid)
            if pooled is not None:
                self._store_value(sid, pooled)
            self.n_insert += 1
        self.insert_seconds += time.perf_counter() - t0
        return sid

    def insert_batch(self, sessions: Sequence[Sequence[int]],
                     pooled: Optional[np.ndarray] = None) -> None:
        for i, s in enumerate(sessions):
            self.insert(s, None if pooled is None else pooled[i])

    def _store_value(self, sid: int, vec: np.ndarray) -> None:
        if self.values is None:
            n = max(self.cfg.value_dim, 1)
            self.values = np.zeros((max(1024, sid + 1), n), dtype=np.int8)
            self._row_scale = np.ones(max(1024, sid + 1), dtype=np.float32)
        if sid >= len(self.values):                    # geometric growth
            old_values, old_scales = self.values, self._row_scale
            grow = np.zeros((max(len(self.values) * 2, sid + 1),
                             self.values.shape[1]), dtype=np.int8)
            grow[:len(old_values)] = old_values
            self.values = grow
            rs = np.ones(len(self.values), dtype=np.float32)
            rs[:len(old_scales)] = old_scales
            self._row_scale = rs
        v = np.asarray(vec, dtype=np.float32)
        scale = np.maximum(np.abs(v).max() / 127.0, 1e-8)
        self.values[sid] = np.clip(np.round(v / scale), -127, 127)\
            .astype(np.int8)
        self._row_scale[sid] = scale

    # ------------------------------------------------------------------- read
    def freeze(self) -> None:
        """Publish an immutable posting block without blocking readers.

        The mutable tail is snapshotted under a short metadata lock. Sorting
        and merging happen outside the lock while readers continue to see the
        previous frozen snapshot plus the complete tail. A second short lock
        publishes the merged array and removes only the snapshot prefix from
        the tail; entries appended during the merge remain mutable.
        """
        if self.cfg.bucket_layout not in ("sorted", "packed"):
            with self._lock:
                if self.cfg.compact_item_sets and self.item_sets:
                    self._compact_all_item_sets()
                if self.cfg.discard_signatures:
                    self.sigs = []
                self._dirty = False
            return

        with self._lock:
            if not self._dirty:
                tail_snapshot = []
                tail_n = 0
                first_sid = 0
            else:
                tail_snapshot = [tuple(band) for band in self._band_buf]
                tail_n = len(tail_snapshot[0]) if tail_snapshot else 0
                first_sid = self.n_insert - tail_n

        if tail_n:
            arrs = [np.fromiter(band, dtype=np.uint64, count=len(band))
                    for band in tail_snapshot]
            new_postings = np.concatenate(arrs)
            new_postings.sort()

            with self._lock:
                if self.posting_index:
                    new_postings = np.concatenate(
                        [self.posting_index[0], new_postings])
                    new_postings.sort()
                self.posting_index = [new_postings]
                self.posting_blocks.append(None)
                self.block_first_sid.append(first_sid)
                self.block_n_sessions.append(tail_n)
                self.bands = self.posting_index
                self._band_buf = [list(band[tail_n:])
                                  for band in self._band_buf]
                self._dirty = any(len(band) for band in self._band_buf)

        with self._lock:
            if self.cfg.compact_item_sets and self.item_sets:
                self._compact_all_item_sets()
            if self.cfg.discard_signatures:
                self.sigs = []

    def _compact_all_item_sets(self) -> None:
        """Append mutable-tail item sets to the global CSR arrays."""
        tail = self.item_sets
        tail_lens = np.fromiter((len(x) for x in tail), dtype=np.int64,
                                count=len(tail))
        if self.item_indptr is None or self.item_ids is None:
            old_ptr = np.zeros(1, dtype=np.int64)
            old_ids = np.empty(0, dtype=np.int32)
        else:
            old_ptr, old_ids = self.item_indptr, self.item_ids
        old_n = len(old_ptr) - 1
        new_ptr = np.empty(old_n + len(tail) + 1, dtype=np.int64)
        new_ptr[:old_n + 1] = old_ptr
        if len(tail):
            np.cumsum(tail_lens, out=new_ptr[old_n + 1:])
            new_ptr[old_n + 1:] += old_ptr[-1]
        new_ids = np.empty(int(new_ptr[-1]), dtype=np.int32)
        new_ids[:len(old_ids)] = old_ids
        pos = len(old_ids)
        for items in tail:
            vals = np.fromiter(items, dtype=np.int32, count=len(items))
            new_ids[pos:pos + len(vals)] = vals
            pos += len(vals)
        self.item_indptr, self.item_ids = new_ptr, new_ids
        self.item_sets = []

    def _candidates(self, items: Sequence[int]) -> List[int]:
        # Snapshot immutable postings and the mutable tail under a short
        # metadata lock, then search without holding the lock.  Freeze can
        # therefore sort/publish a block concurrently with candidate search.
        with self._lock:
            posting_arr = (self.posting_index[0]
                          if self.posting_index else None)
            tail = tuple(tuple(band) for band in
                         (self._band_buf or []))
            deleted = set(self._deleted_sessions)

        sig = minhash_signature(items, self.cfg.sig_len, self.cfg.seed,
                                self._ha, self._hb)
        sig_b, row = sig.tobytes(), self.rows
        cand: Dict[int, int] = {}
        band_ids = np.arange(self.cfg.bands, dtype=np.uint64)
        keys = np.fromiter(
            (_stable_band_key(sig_b[bix * 4 * row:
                                   (bix + 1) * 4 * row])
             for bix in range(self.cfg.bands)),
            dtype=np.uint64, count=self.cfg.bands)
        targets = (band_ids << 56) | (keys << np.uint64(32))

        if posting_arr is not None:
            lows = posting_arr.searchsorted(targets, side="left")
            highs = posting_arr.searchsorted(
                targets + np.uint64(1 << 32), side="right")
            hits = np.concatenate([
                posting_arr[lo:hi]
                for lo, hi in zip(lows.tolist(), highs.tolist())
            ])
            sids = (hits & np.uint64(0xFFFFFFFF)).astype(np.int64)
            if deleted:
                sids = sids[~np.isin(sids, np.fromiter(
                    deleted, dtype=np.int64, count=len(deleted)))]
            if sids.size:
                unique, counts = np.unique(sids, return_counts=True)
                for sid, count in zip(unique.tolist(), counts.tolist()):
                    cand[sid] = cand.get(sid, 0) + int(count)

        if len(tail) == self.cfg.bands:
            for bix in range(self.cfg.bands):
                target = int(targets[bix])
                for coded in tail[bix]:
                    if target <= coded < target + (1 << 32):
                        sid = coded & 0xFFFFFFFF
                        if sid not in deleted:
                            cand[sid] = cand.get(sid, 0) + 1

        if not self.posting_index and not tail:
            # Compatibility with the dictionary layout.
            for bix in range(self.cfg.bands):
                key = (bix, row,
                       sig_b[bix * 4 * row:(bix + 1) * 4 * row])
                for sid in self.buckets.get(key, ()):
                    if sid not in deleted:
                        cand[sid] = cand.get(sid, 0) + 1

        self.probe_ops += len(cand)
        return sorted(cand, key=lambda x: -cand[x])[:self.cfg.max_cand]

    def search(self, items: Sequence[int], k: Optional[int] = None,
               return_values: bool = False):
        """Approximate top-K by true Jaccard among band candidates.

        Returns (neighbor_ids[, dequantized_profiles]).
        """
        k = k or self.cfg.k_r
        s = set(int(i) for i in items)
        cands = self._candidates(items)
        scored = []
        if self.cfg.rank == "profile" and self.values is not None:
            q = self.profile(cands)                      # (C, d)
            qn = q / np.linalg.norm(q, axis=1, keepdims=True).clip(1e-8)
            sims = qn @ qn[0]                            # vs anchor profile
            scored = sorted(zip(sims.tolist(), cands), reverse=True)
        else:
            for sid in cands:
                t = self.item_set(sid)
                inter = len(s & t)
                if inter:
                    scored.append((inter / (len(s) + len(t) - inter), sid))
            scored.sort(reverse=True)
        ids = [sid for _, sid in scored[:k]]
        while len(ids) < k:                     # pad with self (as baseline)
            ids.append(ids[0] if ids else 0)
        if return_values:
            # Return dequantized profiles, not raw int8 codes.  Callers use
            # this path as a model-facing value store.
            return ids, self.profile(ids)
        return ids

    def profile(self, ids: Sequence[int]) -> np.ndarray:
        """Dequantized value rows for batch-time r0 assembly."""
        ids = np.asarray(ids)
        return self.values[ids].astype(np.float32) * \
            self._row_scale[ids][:, None]

    def delete_session(self, sid: int) -> bool:
        """Logical deletion: hide a session atomically from future queries."""
        with self._lock:
            if not (0 <= sid < self.n_insert) or sid in self._deleted_sessions:
                return False
            self._deleted_sessions.add(int(sid))
            return True

    def compact_postings(self) -> int:
        """Remove logically deleted sids from frozen posting blocks.

        CSR item sets and dense value rows remain until a full block
        rebuild; this first stage reclaims posting storage, which is the
        dominant metadata component.
        """
        with self._lock:
            reclaimed = 0
            deleted = (np.fromiter(self._deleted_sessions,
                                   dtype=np.uint64,
                                   count=len(self._deleted_sessions))
                       if self._deleted_sessions else
                       np.empty(0, dtype=np.uint64))
            for bix, arr in enumerate(self.posting_index):
                keep = ~np.isin(arr & 0xFFFFFFFF, deleted)
                reclaimed += arr.nbytes - arr[keep].nbytes
                self.posting_index[bix] = arr[keep]
            self._deleted_sessions.clear()
            return reclaimed

    def item_set(self, sid: int) -> set:
        """Return one item set; CSR mode avoids per-session Python sets."""
        if sid in self._deleted_sessions:
            return set()
        if self.item_indptr is None:
            return self.item_sets[sid] if sid < len(self.item_sets) else set()
        if sid < 0 or sid >= len(self.item_indptr) - 1:
            return set()
        return set(self.item_ids[self.item_indptr[sid]:
                                 self.item_indptr[sid + 1]].tolist())

    def persist(self, path: Path) -> None:
        """Atomically persist a frozen store with stable posting keys.

        The mutable tail must be frozen first.  A temporary compressed NPZ
        file is flushed and renamed, so readers observe either the old or the
        new complete snapshot, never a partially written block.
        """
        if self._dirty or any(len(x) for x in (self._band_buf or [])):
            raise RuntimeError("freeze the mutable tail before persisting")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".",
                                         suffix=".npz", dir=path.parent)
        os.close(fd)
        try:
            metadata = {
                "config": self.cfg.__dict__,
                "n_insert": self.n_insert,
                "block_first_sid": self.block_first_sid,
                "block_n_sessions": self.block_n_sessions,
            }
            np.savez_compressed(
                tmp_name,
                metadata=np.asarray([json.dumps(metadata)]),
                postings=(self.posting_index[0]
                          if self.posting_index else
                          np.empty(0, dtype=np.uint64)),
                item_indptr=(self.item_indptr if self.item_indptr is not None
                             else np.empty(0, dtype=np.int64)),
                item_ids=(self.item_ids if self.item_ids is not None
                          else np.empty(0, dtype=np.int32)),
                values=(self.values if self.values is not None
                        else np.empty((0, self.cfg.value_dim), dtype=np.int8)),
                row_scale=(self._row_scale if self._row_scale is not None
                           else np.empty(0, dtype=np.float32)),
            )
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    @classmethod
    def load(cls, path: Path) -> "IGMS":
        """Load a frozen store; stable keys make blocks process-portable."""
        with np.load(Path(path), allow_pickle=False) as archive:
            cfg_dict = json.loads(str(archive["metadata"][0]))["config"]
            n_insert = json.loads(str(archive["metadata"][0]))["n_insert"]
            meta = json.loads(str(archive["metadata"][0]))
            obj = cls(IGMSConfig(**cfg_dict))
            postings = archive["postings"]
            if postings.size:
                obj.posting_index = [postings]
                obj.bands = [postings]
            obj.block_first_sid = meta["block_first_sid"]
            obj.block_n_sessions = meta["block_n_sessions"]
            obj.posting_blocks = [None] * len(obj.block_first_sid)
            obj.item_indptr = (archive["item_indptr"]
                               if archive["item_indptr"].size else None)
            obj.item_ids = (archive["item_ids"]
                            if archive["item_ids"].size else None)
            obj.values = (archive["values"]
                          if archive["values"].size else None)
            obj._row_scale = (archive["row_scale"]
                              if archive["row_scale"].size else None)
            obj.n_insert = int(n_insert)
            return obj

    # -------------------------------------------------------------- reporting
    def memory_bytes(self) -> Dict[str, int]:
        sig = sum(x.nbytes for x in self.sigs)
        if self.cfg.bucket_layout in ("sorted", "packed"):
            buckets = sum(a.nbytes for a in self.posting_index)
            if self._band_buf is not None:
                buckets += sum(sys.getsizeof(x) for band in self._band_buf
                               for x in band)
            if self.bands is not None and not self.posting_index:
                buckets += sum(a.nbytes for a in self.bands)
        else:
            buckets = sum(len(v) for v in self.buckets.values()) * 8 \
                + len(self.buckets) * 64
        sets = sum(len(s) * 8 for s in self.item_sets)
        compact_sets = 0 if self.item_indptr is None else \
            self.item_indptr.nbytes + self.item_ids.nbytes
        vals = 0 if self.values is None else \
            self.n_insert * self.values.shape[1] * 1
        sets = sets if self.cfg.keep_item_sets else 0
        return {"signatures": sig, "buckets": buckets,
                "item_sets": sets, "compact_item_sets": compact_sets,
                "values": vals,
                "deletion_vector": len(self._deleted_sessions) * 56,
                "total_lean": (sig + buckets + compact_sets + vals
                               + len(self._deleted_sessions) * 56),
                "total": (sig + buckets + sets + compact_sets + vals
                          + len(self._deleted_sessions) * 56)}


# ------------------------------------------------------------------ ground truth
class ExactJaccard:
    """Full-scan exact top-K over the current corpus (evaluation only).

    Uses an item->sessions inverted posting list; overlap counting is
    vectorized with numpy per anchor.
    """

    def __init__(self):
        self.postings: Dict[int, np.ndarray] = {}
        self._buf: Dict[int, List[int]] = {}
        self.corpus_sets: List[set] = []
        self.n = 0

    def insert(self, items: Sequence[int]) -> None:
        sid = self.n
        s = set(int(i) for i in items)
        self.corpus_sets.append(s)
        for i in s:
            self._buf.setdefault(i, []).append(sid)
        self.n += 1

    def finalize(self) -> None:
        self.postings = {i: np.asarray(v, dtype=np.int64)
                         for i, v in self._buf.items()}
        self._buf = {}

    def topk(self, items: Sequence[int], k: int) -> List[int]:
        s = set(int(i) for i in items)
        posts = [self.postings[i] for i in s if i in self.postings]
        if not posts:
            return [0] * k
        counts = np.bincount(np.concatenate(posts), minlength=self.n)
        cand = np.nonzero(counts)[0]
        inter = counts[cand]
        sizes = np.fromiter((len(self.corpus_sets[c]) for c in cand),
                            dtype=np.float64, count=len(cand))
        j = inter / (len(s) + sizes - inter)
        order = np.argsort(-j)[:k]
        return cand[order].tolist()
