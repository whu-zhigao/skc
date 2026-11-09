"""A two-tier posting store with an explicit resident-byte cap.

The store keeps the same MinHash/CSR semantics as IGMS but partitions data
into independently readable blocks.  Each block has:

* one sorted posting array per band;
* length-prefixed CSR item records for exact Jaccard re-ranking;
* optional int8 profile rows and fp32 scales.

A cold block is read with bounded ``np.fromfile`` calls: only the requested
band buckets and candidate item records enter Python memory.  A hot block is
fully resident and counts every posting, offset, item, code, and scale byte
against the configured budget.  Thus total application-owned resident data
has a hard cap, while cold blocks remain searchable on disk.
"""
from __future__ import annotations

from collections import OrderedDict, Counter
import json
import mmap
import os
from pathlib import Path
import threading
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) \
        & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) \
        & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def _stable_key(chunk: bytes) -> int:
    return _splitmix64(int.from_bytes(chunk, "little")) & 0xFFFFFF


def minhash_bands(items: Sequence[int], bands: int, rows: int,
                  ha: np.ndarray, hb: np.ndarray) -> np.ndarray:
    values = np.asarray(sorted(set(int(x) for x in items)), dtype=np.int64)
    if not values.size:
        values = np.asarray([0], dtype=np.int64)
    signature = (((values[None, :] * ha[:, None] + hb[:, None])
                  % 4294967311) & 0xFFFFFFFF).min(axis=1)
    sig = signature.astype(np.uint32).tobytes()
    return np.asarray([
        (band << 56) | (_stable_key(sig[band * 4 * rows:
                                          (band + 1) * 4 * rows]) << 32)
        for band in range(bands)
    ], dtype=np.uint64)


class _ColdBlock:
    def __init__(self, root: Path, block_id: int):
        self.root = root
        self.block_id = block_id
        meta = json.loads((root / "meta.json").read_text())
        self.meta = meta["blocks"][str(block_id)]
        self.posting_path = root / f"block-{block_id}.postings.bin"
        self.bucket_index_path = root / f"block-{block_id}.bucket-index.bin"
        self.item_path = root / f"block-{block_id}.items.bin"
        self.profile_path = root / f"block-{block_id}.profiles.npy"
        self.scale_path = root / f"block-{block_id}.scales.npy"

class _HotBlock:
    def __init__(self, root: Path, block_id: int):
        cold = _ColdBlock(root, block_id)
        self.block_id = block_id
        self.first_sid = cold.meta["first_session"]
        self.n_sessions = cold.meta["n_sessions"]
        with cold.posting_path.open("rb") as file:
            self.postings = np.fromfile(file, dtype=np.uint64)
        self.band_offsets = np.asarray(cold.meta["band_offsets"],
                                       dtype=np.int64)
        self.band_counts = np.asarray(cold.meta["band_counts"],
                                      dtype=np.int64)
        self.item_offsets = np.load(root / f"block-{block_id}.offsets.npy",
                                    mmap_mode="r")
        with cold.item_path.open("rb") as file:
            self.items = np.fromfile(file, dtype=np.int32)
        self.profiles = (np.load(cold.profile_path, mmap_mode="r")
                         if cold.profile_path.exists() else None)
        self.scales = (np.load(cold.scale_path, mmap_mode="r")
                       if cold.scale_path.exists() else None)

    def bytes(self) -> int:
        total = (self.postings.nbytes + self.band_offsets.nbytes +
                 self.band_counts.nbytes + self.item_offsets.nbytes +
                 self.items.nbytes)
        if self.profiles is not None:
            total += self.profiles.nbytes
        if self.scales is not None:
            total += self.scales.nbytes
        return int(total)


class BoundedTieredPostingStore:
    """MinHash posting search with hard hot-memory cap and cold fallback."""

    def __init__(self, root: Path, memory_budget_bytes: int,
                 bands: int = 32, rows: int = 2, seed: int = 0,
                 k: int = 10, max_candidates: int = 512,
                 direct_io: bool = False,
                 evict_after_read: bool = False):
        self.root = Path(root)
        self.memory_budget_bytes = int(memory_budget_bytes)
        self.bands = bands
        self.rows = rows
        self.k = k
        self.max_candidates = max_candidates
        self.direct_io = bool(direct_io)
        self.evict_after_read = bool(evict_after_read)
        meta = json.loads((self.root / "meta.json").read_text())
        self.meta = meta
        rng = np.random.RandomState(seed)
        sig_len = bands * rows
        self.ha = rng.randint(1, (1 << 31) - 1, size=sig_len).astype(np.int64)
        self.hb = rng.randint(0, (1 << 31) - 1, size=sig_len).astype(np.int64)
        self.block_ids = [int(x) for x in meta["blocks"].keys()]
        self._hot: OrderedDict[int, _HotBlock] = OrderedDict()
        self._block_scores: Counter = Counter()
        self._cold = {bid: _ColdBlock(self.root, bid)
                      for bid in self.block_ids}
        self.disk_bucket_reads = 0
        self.disk_item_reads = 0
        self.disk_bucket_bytes = 0
        self.disk_item_bytes = 0
        self.disk_index_reads = 0
        self._bucket_maps: Dict[int, mmap.mmap] = {}
        self.temporary_bytes_max = 0
        self._lock = threading.RLock()

    @property
    def resident_bytes(self) -> int:
        return sum(block.bytes() for block in self._hot.values())

    def _account_temp(self, arrays: Iterable[np.ndarray]) -> None:
        size = sum(arr.nbytes for arr in arrays)
        self.temporary_bytes_max = max(self.temporary_bytes_max, int(size))

    def _promote_locked(self, block_id: int) -> None:
        if block_id in self._hot:
            self._hot.move_to_end(block_id)
            return
        candidate = _HotBlock(self.root, block_id)
        self._hot[block_id] = candidate
        while self.resident_bytes > self.memory_budget_bytes:
            victim_id, victim = next(iter(self._hot.items()))
            del self._hot[victim_id]

    def enforce_budget(self) -> None:
        """Evict coldest hot blocks until the configured cap is respected."""
        with self._lock:
            while self._hot and self.resident_bytes > self.memory_budget_bytes:
                victim_id = next(iter(self._hot))
                del self._hot[victim_id]

    def _find_bucket(self, cold: _ColdBlock, band: int,
                     idx_off: int, idx_count: int,
                     target_key: np.uint64) -> Tuple[int, int]:
        """Binary search a fixed-width memory-mapped sparse bucket index.

        The mapping is a cold-storage access mechanism, not accounted resident
        cache-plane state.  Pages may enter the OS cache, exactly as ordinary
        buffered reads may; only application-owned arrays are counted.
        """
        with self._lock:
            mapping = self._bucket_maps.get(cold.block_id)
            if mapping is None:
                with cold.bucket_index_path.open("rb") as file:
                    mapping = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
                self._bucket_maps[cold.block_id] = mapping

        def record(mid: int) -> Tuple[int, int]:
            off = (idx_off + mid) * 16
            first = int.from_bytes(mapping[off:off + 8], "little")
            second = int.from_bytes(mapping[off + 8:off + 16], "little")
            self.disk_index_reads += 1
            self.disk_bucket_bytes += 16
            return (first >> 32) & 0xFFFFFFFF, second

        lo, hi = 0, idx_count
        found_at = -1
        while lo < hi:
            mid = (lo + hi) // 2
            key, packed = record(mid)
            if key == int(target_key):
                found_at = mid
                break
            if key < int(target_key):
                lo = mid + 1
            else:
                hi = mid
        if found_at < 0:
            return 0, 0
        _key, packed = record(found_at)
        if self.evict_after_read:
            try:
                mapping.madvise(mmap.MADV_DONTNEED)
            except (AttributeError, OSError):
                pass
        return packed >> 32, packed & 0xFFFFFFFF

    def _open_cold(self, path: Path):
        if self.direct_io and hasattr(os, "O_DIRECT"):
            flags = os.O_RDONLY | os.O_DIRECT
            return os.fdopen(os.open(path, flags), "rb"), True
        return path.open("rb"), True

    def _cold_bucket(self, block_id: int, band: int,
                     target: int) -> np.ndarray:
        cold = self._cold[block_id]
        offset = int(cold.meta["band_offsets"][band])
        count = int(cold.meta["band_counts"][band])
        if count == 0:
            return np.empty(0, dtype=np.uint64)
        target_key = np.uint64((np.uint64(target) >> np.uint64(32)) & np.uint64(0xFFFFFFFF))
        posting_start, posting_count = offset, count

        # Sparse (key, start, count) indexes are built for every frozen block.
        # Read only that index slice, locate the exact posting bucket, and then
        # read only the matching postings rather than the whole band array.
        bucket_offsets = cold.meta.get("bucket_index_offsets")
        if bucket_offsets is not None:
            idx_off = int(bucket_offsets[band])
            idx_count = int(bucket_offsets[band + 1] - idx_off)
            if idx_count:
                rel_start, run_count = self._find_bucket(
                    cold, band, idx_off, idx_count, target_key
                )
                posting_start = offset + rel_start
                posting_count = run_count

        if posting_count == 0:
            return np.empty(0, dtype=np.uint64)
        file, owns_fd = self._open_cold(cold.posting_path)
        try:
            file.seek(posting_start * 8)
            arr = np.fromfile(file, dtype=np.uint64, count=posting_count)
            self.disk_bucket_bytes += arr.nbytes
            if owns_fd and self.evict_after_read:
                try:
                    os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                except (AttributeError, OSError):
                    pass
        finally:
            file.close()
        self.disk_bucket_reads += 1
        self._account_temp([arr])
        return arr

    def _cold_items(self, block_id: int, sids: Sequence[int]) -> List[set]:
        cold = self._cold[block_id]
        out = []
        with cold.item_path.open("rb") as file:
            for sid in sids:
                local = int(sid) - cold.meta["first_session"]
                begin = int(cold.meta["item_offsets"][local])
                count = int(cold.meta["item_offsets"][local + 1] - begin)
                file.seek(begin * 4)
                values = np.fromfile(file, dtype=np.int32, count=count)
                self.disk_item_reads += 1
                self.disk_item_bytes += values.nbytes
                if self.evict_after_read:
                    try:
                        os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                    except (AttributeError, OSError):
                        pass
                out.append(set(values.tolist()))
        self._account_temp([np.fromiter((len(x) for x in out), dtype=np.int32,
                                         count=len(out))])
        return out

    def _items(self, block_id: int, sids: Sequence[int]) -> List[set]:
        if block_id in self._hot:
            hot = self._hot[block_id]
            first = hot.first_sid
            return [set(hot.items[int(hot.item_offsets[int(sid) - first]):
                                  int(hot.item_offsets[int(sid) - first + 1])]
                        .tolist()) for sid in sids]
        return self._cold_items(block_id, sids)

    def search(self, items: Sequence[int], return_profiles: bool = False):
        query_bands = minhash_bands(items, self.bands, self.rows,
                                    self.ha, self.hb)
        candidates: Dict[int, int] = {}
        contributing_blocks = set()
        with self._lock:
            hot_snapshot = dict(self._hot)
        for block_id in self.block_ids:
            first = self._cold[block_id].meta["first_session"]
            n_sessions = self._cold[block_id].meta["n_sessions"]
            block_sids = set()
            for band in range(self.bands):
                target = int(query_bands[band])
                if block_id in hot_snapshot:
                    hot = hot_snapshot[block_id]
                    off = int(hot.band_offsets[band])
                    cnt = int(hot.band_counts[band])
                    arr = hot.postings[off:off + cnt]
                    low = int(np.searchsorted(arr, target, side="left"))
                    high = int(np.searchsorted(
                        arr, target + (1 << 32), side="right"))
                    hits = arr[low:high]
                else:
                    hits = self._cold_bucket(block_id, band, target)
                sids = (hits.astype(np.int64) &
                        np.int64(0xFFFFFFFF)).tolist()
                for sid in sids:
                    candidates[sid] = candidates.get(sid, 0) + 1
                    block_sids.add(sid)
            if block_sids:
                contributing_blocks.add(block_id)

        ranked = sorted(candidates.items(), key=lambda x: (-x[1], x[0]))
        ranked = ranked[:self.max_candidates]
        query_set = set(int(x) for x in items)
        scored = []
        by_block: Dict[int, List[int]] = {}
        for sid, _ in ranked:
            by_block.setdefault(sid * 0 + int(sid) // 1000000000, []).append(sid)
        # Block ids are recovered from cold metadata rather than encoded in
        # the 64-bit posting word.
        by_block.clear()
        bounds = []
        for bid in self.block_ids:
            first = self._cold[bid].meta["first_session"]
            end = first + self._cold[bid].meta["n_sessions"]
            bounds.append((first, end, bid))
        for sid, _ in ranked:
            for first, end, bid in bounds:
                if first <= sid < end:
                    by_block.setdefault(bid, []).append(sid)
                    break
        for bid, sids in by_block.items():
            item_sets = self._items(bid, sids)
            for sid, items2 in zip(sids, item_sets):
                inter = len(query_set & items2)
                if inter:
                    denom = len(query_set) + len(items2) - inter
                    scored.append((inter / max(denom, 1), int(sid)))
        scored.sort(reverse=True)
        ids = [sid for _, sid in scored[:self.k]]

        with self._lock:
            for bid in contributing_blocks:
                self._block_scores[bid] += 1
            # Promote at most one coldest-accessed contributing block. This
            # keeps promotion work bounded and avoids global reconstruction.
            if contributing_blocks:
                cold_contributing = [b for b in contributing_blocks
                                     if b not in self._hot]
                candidates = cold_contributing or list(contributing_blocks)
                promote = max(candidates,
                              key=lambda b: (self._block_scores[b], -b))
                self._promote_locked(promote)
            self.enforce_budget()

        if not return_profiles:
            return ids
        profiles = []
        for sid in ids:
            for first, end, bid in bounds:
                if first <= sid < end:
                    if bid in self._hot:
                        hot = self._hot[bid]
                        local = sid - hot.first_sid
                        profiles.append(hot.profiles[local] *
                                        hot.scales[local])
                    else:
                        cold = self._cold[bid]
                        local = sid - cold.meta["first_session"]
                        codes = np.load(cold.profile_path, mmap_mode="r")[local]
                        scale = np.load(cold.scale_path, mmap_mode="r")[local]
                        profiles.append(codes.astype(np.float32) * scale)
                    break
        return ids, np.asarray(profiles, dtype=np.float32)


def build_bounded_tier(sessions: Sequence[Sequence[int]],
                       profiles: Optional[np.ndarray],
                       root: Path, block_sessions: int = 8192,
                       bands: int = 32, rows: int = 2, seed: int = 0,
                       hot_blocks: int = 1) -> BoundedTieredPostingStore:
    """Materialize cold blocks and return a capped hot resident view."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    sig_len = bands * rows
    ha = rng.randint(1, (1 << 31) - 1, size=sig_len).astype(np.int64)
    hb = rng.randint(0, (1 << 31) - 1, size=sig_len).astype(np.int64)
    block_meta = {}

    for b, start in enumerate(range(0, len(sessions), block_sessions)):
        chunk = sessions[start:start + block_sessions]
        band_arrays = [[] for _ in range(bands)]
        item_offsets = [0]
        flat_items = []
        for local, session in enumerate(chunk):
            sig = minhash_bands(session, bands, rows, ha, hb)
            for band in range(bands):
                band_arrays[band].append(sig[band] | np.uint64(start + local))
            items = sorted(set(int(x) for x in session))
            flat_items.extend(items)
            item_offsets.append(len(flat_items))
        posting_parts = []
        offsets = []
        counts = []
        bucket_index_parts = []
        bucket_index_offsets = [0]
        bucket_index_count = 0
        posting_offset = 0
        for band, values in enumerate(band_arrays):
            arr = np.asarray(values, dtype=np.uint64)
            arr.sort()
            offsets.append(posting_offset)
            posting_offset += len(arr)
            counts.append(len(arr))
            posting_parts.append(arr)
            if arr.size:
                key32 = (arr >> np.uint64(32)) & np.uint64(0xFFFFFFFF)
                change = np.r_[True, key32[1:] != key32[:-1]]
                starts = np.flatnonzero(change).astype(np.uint64)
                unique_keys = key32[change].astype(np.uint64)
                run_counts = np.diff(np.r_[starts, np.uint64(arr.size)]).astype(np.uint64)
                packed = (starts << np.uint64(32)) | run_counts
                key_records = unique_keys << np.uint64(32)
                bucket_index_parts.append(np.stack([key_records, packed], axis=1).reshape(-1))
                bucket_index_count += len(unique_keys)
            bucket_index_offsets.append(bucket_index_count)
        postings = np.concatenate(posting_parts)
        postings.tofile(root / f"block-{b}.postings.bin")
        if bucket_index_parts:
            np.concatenate(bucket_index_parts).tofile(root / f"block-{b}.bucket-index.bin")
        else:
            np.empty(0, dtype=np.uint64).tofile(root / f"block-{b}.bucket-index.bin")
        np.save(root / f"block-{b}.offsets.npy",
                np.asarray(item_offsets, dtype=np.int64))
        np.asarray(flat_items, dtype=np.int32).tofile(
            root / f"block-{b}.items.bin")
        if profiles is not None:
            block_profiles = profiles[start:start + len(chunk)]
            scales = np.maximum(
                np.linalg.norm(block_profiles, axis=1) /
                np.sqrt(block_profiles.shape[1]) / 127.0, 1e-8
            ).astype(np.float32)
            # Per-coordinate symmetric int8 quantization using a global row
            # scale is easier to read from cold storage and remains bounded.
            max_abs = np.maximum(np.abs(block_profiles).max(axis=1), 1e-8)
            scales = max_abs / 127.0
            codes = np.clip(np.rint(block_profiles / scales[:, None]),
                            -127, 127).astype(np.int8)
            np.save(root / f"block-{b}.profiles.npy", codes)
            np.save(root / f"block-{b}.scales.npy", scales)
        block_meta[str(b)] = {
            "first_session": start,
            "n_sessions": len(chunk),
            "band_offsets": offsets,
            "band_counts": counts,
            "bucket_index_offsets": bucket_index_offsets,
            "item_offsets": item_offsets,
        }

    meta = {
        "n_sessions": len(sessions),
        "block_sessions": block_sessions,
        "bands": bands,
        "rows": rows,
        "seed": seed,
        "blocks": block_meta,
    }
    (root / "meta.json").write_text(json.dumps(meta))
    store = BoundedTieredPostingStore(
        root, memory_budget_bytes=1 << 60, bands=bands, rows=rows,
        seed=seed,
    )
    # Initially resident blocks are the first ones; subsequent promotion is
    # based on observed contribution. An offline builder can instead provide
    # NFA block scores before serving.
    for bid in range(min(hot_blocks, len(block_meta))):
        store._promote_locked(bid)
    return store
