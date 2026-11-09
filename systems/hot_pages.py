"""Packed, bounded hot pages for model-derived recommendation profiles.

The store deliberately separates three concerns: bounded residency, immutable
publication, and page-local drift accounting.  Replacement pages are copied
outside the metadata lock and published atomically.  An optional int8 value
plane exercises the full ``drift + quantization`` dispersion contract rather
than storing dequantized fp32 rows only.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .source_snapshot import SourceSnapshot, as_snapshot


class HotPageDirtyError(RuntimeError):
    """One or more requested packed pages crossed the drift threshold."""


@dataclass(frozen=True)
class PackedPageView:
    page_id: int
    source_ids: Tuple[int, ...]
    profiles: np.ndarray
    scales: Optional[np.ndarray] = None
    drift_bound: float = 0.0
    quant_bound: float = 0.0
    version: int = 0
    dirty: bool = False
    model_epoch: int = 0

    def __post_init__(self) -> None:
        if self.profiles.ndim != 2:
            raise ValueError("profiles must be 2-D")
        if len(self.source_ids) != len(self.profiles):
            raise ValueError("ids and profiles have different lengths")
        if self.scales is not None and len(self.scales) != len(self.profiles):
            raise ValueError("ids and scales have different lengths")
        self.profiles.flags.writeable = False
        if self.scales is not None:
            self.scales.flags.writeable = False

    def row(self, row: int) -> np.ndarray:
        if self.scales is None:
            return self.profiles[row]
        return self.profiles[row].astype(np.float32, copy=False) * self.scales[row]

    @property
    def dispersion_bound(self) -> float:
        return float(self.drift_bound + self.quant_bound)


class BoundedPackedPageStore:
    """A hard-capped set of immutable, compact profile pages."""

    def __init__(
        self,
        memory_budget_bytes: int,
        dimension: int,
        page_size: int = 128,
        theta_max: float = 0.1,
        quantize: bool = False,
        quant_bits: int = 8,
    ) -> None:
        if memory_budget_bytes <= 0 or dimension <= 0 or page_size <= 0:
            raise ValueError("budget, dimension, and page size must be positive")
        if quantize and quant_bits != 8:
            raise ValueError("the reference int8 value plane requires quant_bits=8")
        self.memory_budget_bytes = int(memory_budget_bytes)
        self.dimension = int(dimension)
        self.page_size = int(page_size)
        self.theta_max = float(theta_max)
        self.quantize = bool(quantize)
        self.quant_bits = int(quant_bits)
        self.pages: Dict[int, PackedPageView] = {}
        self.page_items: Dict[int, np.ndarray] = {}
        self.ledger: Dict[int, np.ndarray] = {}
        self.version: Dict[int, int] = {}
        self.dirty: set[int] = set()
        self.refresh_baseline: Dict[int, np.ndarray] = {}
        self.refresh_version: Dict[int, int] = {}
        self._refresh_rows: Dict[int, np.ndarray] = {}
        self.refresh_epoch = 0
        self.epoch = 0
        self.resident_sids: Optional[np.ndarray] = None
        self.resident_page: Optional[np.ndarray] = None
        self.item_total: Dict[int, float] = {}
        self.model_epoch = 0

        # CSR-style item -> (page, ledger position) index.  Without this index
        # every model update scans all page memberships; with it update work is
        # proportional to the number of affected (item, page) pairs.
        self.item_index_items: Optional[np.ndarray] = None
        self.item_index_offsets: Optional[np.ndarray] = None
        self.item_index_pages: Optional[np.ndarray] = None
        self.item_index_positions: Optional[np.ndarray] = None
        self._lock = threading.RLock()

    @staticmethod
    def _stored_profile_bytes(
        n_rows: int, dimension: int, quantized: bool
    ) -> int:
        # Quantized rows are int8 codes plus one fp32 scale.  The legacy
        # reference path stores dequantized fp32 rows plus a scale allowance.
        return n_rows * (dimension * (1 if quantized else 4) + 4)

    def capacity(self, n_member_items: int = 0) -> int:
        """Maximum admitted rows for a known ledger-item allowance."""
        per_row = self._stored_profile_bytes(1, self.dimension, self.quantize) + 12
        ledger_and_index = int(n_member_items) * (8 + 16)
        return max(0, (self.memory_budget_bytes - ledger_and_index) // per_row)

    def _item_index_bytes(self) -> int:
        if self.item_index_offsets is None:
            return 0
        return int(
            self.item_index_items.nbytes
            + self.item_index_offsets.nbytes
            + self.item_index_pages.nbytes
            + self.item_index_positions.nbytes
        )

    def resident_bytes(self) -> int:
        with self._lock:
            stored_profiles = sum(
                page.profiles.nbytes
                + (0 if page.scales is None else page.scales.nbytes)
                for page in self.pages.values()
            )
            metadata_bytes = (
                self.resident_sids.nbytes + self.resident_page.nbytes
                if self.resident_sids is not None
                else 0
            )
            ledger_bytes = sum(arr.nbytes for arr in self.ledger.values())
            membership_bytes = sum(
                arr.nbytes for arr in self.page_items.values()
            )
            return int(
                stored_profiles
                + metadata_bytes
                + ledger_bytes
                + membership_bytes
                + self._item_index_bytes()
            )

    def resident_components(self) -> Dict[str, int]:
        """Exact accounted bytes by component, for provenance and tables."""
        with self._lock:
            profile_bytes = sum(
                page.profiles.nbytes
                + (0 if page.scales is None else page.scales.nbytes)
                for page in self.pages.values()
            )
            metadata_bytes = (
                self.resident_sids.nbytes + self.resident_page.nbytes
                if self.resident_sids is not None
                else 0
            )
            ledger_bytes = sum(arr.nbytes for arr in self.ledger.values())
            membership_bytes = sum(
                arr.nbytes for arr in self.page_items.values()
            )
            return {
                "profiles_and_scales": int(profile_bytes),
                "resident_maps": int(metadata_bytes),
                "ledger": int(ledger_bytes),
                "membership": int(membership_bytes),
                "item_page_index": int(self._item_index_bytes()),
                "total": int(
                    profile_bytes + metadata_bytes + ledger_bytes
                    + membership_bytes + self._item_index_bytes()
                ),
            }

    def _encode_rows(self, rows: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        values = np.asarray(rows, dtype=np.float32)
        if not self.quantize:
            return values.copy(), None, 0.0
        levels = float(2 ** (self.quant_bits - 1) - 1)
        max_abs = np.maximum(np.abs(values).max(axis=1), 1e-12).astype(np.float32)
        scales = (max_abs / levels).astype(np.float32)
        codes = np.clip(
            np.rint(values / scales[:, None]), -levels, levels
        ).astype(np.int8)
        coordinate_errors = (
            np.full((len(scales), values.shape[1]), 0.5, dtype=np.float32)
            * scales[:, None]
        )
        quant_bound = float(
            np.linalg.norm(coordinate_errors, axis=1).max()
        )
        return codes, scales, quant_bound

    def _build_item_index(self, membership: Dict[int, np.ndarray]) -> None:
        if not membership:
            self.item_index_items = np.empty(0, dtype=np.int32)
            self.item_index_offsets = np.zeros(1, dtype=np.int64)
            self.item_index_pages = np.empty(0, dtype=np.int32)
            self.item_index_positions = np.empty(0, dtype=np.int32)
            return
        pairs: List[Tuple[int, int, int]] = []
        for page_id, items in membership.items():
            for position, item in enumerate(items.tolist()):
                pairs.append((int(item), int(page_id), position))
        pairs.sort(key=lambda x: x[0])
        pair_items = np.fromiter((x[0] for x in pairs), dtype=np.int32, count=len(pairs))
        self.item_index_items = np.unique(pair_items)
        rows = np.searchsorted(self.item_index_items, pair_items)
        counts = np.bincount(rows, minlength=len(self.item_index_items))
        self.item_index_offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=self.item_index_offsets[1:])
        self.item_index_pages = np.fromiter(
            (x[1] for x in pairs), dtype=np.int32, count=len(pairs)
        )
        self.item_index_positions = np.fromiter(
            (x[2] for x in pairs), dtype=np.int32, count=len(pairs)
        )

    def admit(
        self,
        source_ids: Sequence[int],
        source_profiles: np.ndarray,
        session_items: Sequence[Iterable[int]],
    ) -> None:
        """Pack admitted sessions and enforce the resident-byte cap."""
        ids = np.asarray(sorted({int(x) for x in source_ids}), dtype=np.int64)
        if not len(ids):
            with self._lock:
                self._clear_locked()
            return
        profiles_all = np.asarray(source_profiles, dtype=np.float32)

        def build(candidate_ids: np.ndarray):
            pages: Dict[int, PackedPageView] = {}
            membership: Dict[int, np.ndarray] = {}
            ledgers: Dict[int, np.ndarray] = {}
            versions: Dict[int, int] = {}
            starts: List[int] = []
            for page_id, start in enumerate(
                range(0, len(candidate_ids), self.page_size)
            ):
                end = min(start + self.page_size, len(candidate_ids))
                rows = np.arange(start, end)
                page_ids_array = candidate_ids[rows]
                items = sorted(
                    {
                        int(item)
                        for sid in page_ids_array.tolist()
                        for item in session_items[int(sid)]
                    }
                )
                membership[page_id] = np.asarray(items, dtype=np.int32)
                # float64 accumulation avoids a contract violation caused by
                # repeated float32 rounding over many small optimizer steps.
                ledgers[page_id] = np.zeros(len(items), dtype=np.float64)
                versions[page_id] = 0
                codes, scales, quant_bound = self._encode_rows(profiles_all[page_ids_array])
                pages[page_id] = PackedPageView(
                    page_id=page_id,
                    source_ids=tuple(int(x) for x in page_ids_array),
                    profiles=codes,
                    scales=scales,
                    drift_bound=0.0,
                    quant_bound=quant_bound,
                    version=0,
                    model_epoch=0,
                )
                starts.extend([page_id] * len(page_ids_array))

            # Account the exact representation that will remain resident.
            profile_bytes = sum(
                page.profiles.nbytes
                + (0 if page.scales is None else page.scales.nbytes)
                for page in pages.values()
            )
            page_map = np.asarray(starts, dtype=np.int32)
            metadata_bytes = candidate_ids.nbytes + page_map.nbytes
            ledger_bytes = sum(arr.nbytes for arr in ledgers.values())
            membership_bytes = sum(arr.nbytes for arr in membership.values())

            # Build a temporary CSR layout to obtain its byte count without
            # retaining Python tuple containers.
            saved = (self.item_index_items, self.item_index_offsets,
                     self.item_index_pages, self.item_index_positions)
            self._build_item_index(membership)
            index_bytes = self._item_index_bytes()
            (self.item_index_items, self.item_index_offsets,
             self.item_index_pages, self.item_index_positions) = saved
            total = profile_bytes + metadata_bytes + ledger_bytes + membership_bytes + index_bytes
            return pages, membership, ledgers, versions, page_map, total

        pages, membership, ledgers, versions, page_map, total = build(ids)
        while len(ids) and total > self.memory_budget_bytes:
            remove = max(1, len(ids) // 50)
            ids = ids[:-remove]
            pages, membership, ledgers, versions, page_map, total = build(ids)
        with self._lock:
            self._clear_locked()
            self.epoch += 1
            self.pages = pages
            self.page_items = membership
            self.ledger = ledgers
            self.version = versions
            self.resident_sids = ids.copy()
            self.resident_page = page_map
            self._build_item_index(membership)

    def _clear_locked(self) -> None:
        self.pages = {}
        self.page_items = {}
        self.ledger = {}
        self.version = {}
        self.dirty = set()
        self.refresh_baseline = {}
        self.refresh_version = {}
        self.refresh_epoch = 0
        self.epoch += 1

    def clone(self) -> "BoundedPackedPageStore":
        """Return a fresh store with the same initially admitted pages.

        Cloning is used by repeated SLA arms so every arm starts from the same
        encoded profile set without rebuilding a full corpus-wide profile
        matrix.  Internal arrays are deep-copied.
        """
        with self._lock:
            clone = BoundedPackedPageStore(
                self.memory_budget_bytes,
                self.dimension,
                page_size=self.page_size,
                theta_max=self.theta_max,
                quantize=self.quantize,
                quant_bits=self.quant_bits,
            )
            clone.pages = {
                pid: PackedPageView(
                    page_id=pid,
                    source_ids=page.source_ids,
                    profiles=page.profiles.copy(),
                    scales=None if page.scales is None else page.scales.copy(),
                    drift_bound=page.drift_bound,
                    quant_bound=page.quant_bound,
                    version=page.version,
                    dirty=page.dirty,
                )
                for pid, page in self.pages.items()
            }
            clone.page_items = {k: v.copy() for k, v in self.page_items.items()}
            clone.ledger = {k: v.copy() for k, v in self.ledger.items()}
            clone.version = dict(self.version)
            clone.dirty = set(self.dirty)
            clone.resident_sids = (
                None if self.resident_sids is None
                else self.resident_sids.copy()
            )
            clone.resident_page = (
                None if self.resident_page is None
                else self.resident_page.copy()
            )
            clone.item_total = dict(self.item_total)
            clone.model_epoch = int(self.model_epoch)
            clone.item_index_items = (
                None if self.item_index_items is None
                else self.item_index_items.copy()
            )
            clone.item_index_offsets = (
                None if self.item_index_offsets is None
                else self.item_index_offsets.copy()
            )
            clone.item_index_pages = (
                None if self.item_index_pages is None
                else self.item_index_pages.copy()
            )
            clone.item_index_positions = (
                None if self.item_index_positions is None
                else self.item_index_positions.copy()
            )
            clone.epoch += 1
            return clone
        self.resident_sids = None
        self.resident_page = None
        self.item_index_items = None
        self.item_index_offsets = None
        self.item_index_pages = None
        self.item_index_positions = None

    def _affected(self, item: int) -> Iterable[Tuple[int, int]]:
        if self.item_index_items is None or self.item_index_offsets is None:
            return ()
        row = int(np.searchsorted(self.item_index_items, item))
        if row >= len(self.item_index_items) or int(self.item_index_items[row]) != item:
            return ()
        begin, end = int(self.item_index_offsets[row]), int(self.item_index_offsets[row + 1])
        return (
            (int(self.item_index_pages[j]), int(self.item_index_positions[j]))
            for j in range(begin, end)
        )

    def update_drift(self, item_drift: Mapping[int, float]) -> None:
        """Bulk-accumulate drift through the CSR item-to-page index.

        The previous implementation performed a Python search and inner loop
        for every drifted item.  This version vectorizes membership lookup and
        updates each touched page once, which is important when a real trace
        moves most of the vocabulary.
        """
        if not item_drift:
            return
        with self._lock:
            self.model_epoch += 1
            model_epoch = int(self.model_epoch)
            item_ids = np.asarray(list(item_drift.keys()), dtype=np.int64)
            values = np.asarray(list(item_drift.values()), dtype=np.float64)
            for item, value in zip(item_ids.tolist(), values.tolist()):
                self.item_total[int(item)] = (
                    self.item_total.get(int(item), 0.0) + float(value)
                )

            if self.item_index_items is None or self.item_index_offsets is None:
                return
            rows = np.searchsorted(self.item_index_items, item_ids)
            valid = (
                (rows < len(self.item_index_items))
                & (self.item_index_items[np.minimum(
                    rows, len(self.item_index_items) - 1
                )] == item_ids)
            )
            rows = rows[valid]
            values = values[valid]
            if not len(rows):
                return

            counts = (
                self.item_index_offsets[rows + 1]
                - self.item_index_offsets[rows]
            )
            index_positions = np.concatenate([
                np.arange(
                    int(self.item_index_offsets[int(row)]),
                    int(self.item_index_offsets[int(row) + 1]),
                    dtype=np.int64,
                )
                for row in rows.tolist()
            ])
            value_repeats = np.repeat(values, counts)
            touched_pages = self.item_index_pages[index_positions]
            ledger_positions = self.item_index_positions[index_positions]

            for page_id in np.unique(touched_pages).tolist():
                mask = touched_pages == page_id
                self.ledger[page_id][ledger_positions[mask]] += (
                    value_repeats[mask]
                )

            # Publish immutable metadata views only for touched pages.
            for page_id in np.unique(touched_pages).tolist():
                old = self.pages[page_id]
                bound = float(self.ledger[page_id].max())
                dirty = bound > self.theta_max
                if dirty:
                    self.dirty.add(page_id)
                else:
                    self.dirty.discard(page_id)
                self.pages[page_id] = PackedPageView(
                    page_id=old.page_id,
                    source_ids=old.source_ids,
                    profiles=old.profiles,
                    scales=old.scales,
                    drift_bound=bound,
                    quant_bound=old.quant_bound,
                    version=old.version,
                    dirty=dirty,
                    model_epoch=model_epoch,
                )

    def dirty_pages(self) -> Tuple[int, ...]:
        with self._lock:
            return tuple(sorted(self.dirty))

    def mark_all_dirty(self) -> int:
        with self._lock:
            for page_id, old in list(self.pages.items()):
                self.dirty.add(page_id)
                self.pages[page_id] = PackedPageView(
                    page_id=old.page_id,
                    source_ids=old.source_ids,
                    profiles=old.profiles,
                    scales=old.scales,
                    drift_bound=max(old.drift_bound, self.theta_max * 1.5),
                    quant_bound=old.quant_bound,
                    version=old.version,
                    dirty=True,
                )
            return len(self.pages)

    def prepare_refresh(
        self,
        page_ids: Optional[Iterable[int]] = None,
        source_profiles: Optional[SourceSnapshot | np.ndarray] = None,
    ) -> Tuple[Tuple[int, ...], int]:
        selected_tuple = tuple(
            sorted(set(page_ids) if page_ids is not None else self.dirty)
        )
        snapshot = as_snapshot(source_profiles) if source_profiles is not None else None
        with self._lock:
            for page_id in selected_tuple:
                if page_id not in self.page_items:
                    raise KeyError(f"unknown packed page {page_id}")
                self.refresh_baseline[page_id] = np.asarray(
                    [self.item_total.get(int(item), 0.0) for item in self.page_items[page_id]],
                    dtype=np.float64,
                )
                self.refresh_version[page_id] = int(self.version[page_id])
                prepared_epoch = int(self.epoch)
                self.refresh_epoch = int(prepared_epoch)
                if snapshot is not None:
                    rows = np.asarray(self.pages[page_id].source_ids, dtype=np.int64)
                    copied = snapshot.profiles[rows].copy()
                    copied.flags.writeable = False
                    self._refresh_rows[page_id] = copied
            return selected_tuple, len(self.item_total)

    def publish_pages(
        self,
        source_profiles: Optional[np.ndarray] = None,
        page_ids: Optional[Iterable[int]] = None,
        source_fn: Optional[Callable[[Sequence[int]], np.ndarray]] = None,
    ) -> int:
        """Copy replacement pages outside the lock, then publish atomically."""
        selected_tuple = tuple(
            sorted(set(page_ids) if page_ids is not None else self.refresh_baseline)
        )
        if source_profiles is not None:
            source_fn = None
            if set(self._refresh_rows) - set(selected_tuple):
                self._refresh_rows.clear()
            snapshot = as_snapshot(source_profiles)
            with self._lock:
                for page_id in selected_tuple:
                    if page_id in self._refresh_rows:
                        continue
                    rows = np.asarray(self.pages[page_id].source_ids, dtype=np.int64)
                    copied = snapshot.profiles[rows].copy()
                    copied.flags.writeable = False
                    self._refresh_rows[page_id] = copied
        if not self._refresh_rows and source_profiles is None and source_fn is None:
            raise ValueError("prepare_refresh must pin a source snapshot or supply a pinned source_fn")

        replacements: List[Tuple[int, np.ndarray, Optional[np.ndarray], float]] = []
        with self._lock:
            prepared_epoch = int(self.refresh_epoch)
            for page_id in selected_tuple:
                if page_id not in self.pages or prepared_epoch != self.epoch:
                    raise KeyError(f"unknown packed page {page_id}")
                old = self.pages[page_id]
                replacements.append(
                    (page_id, np.asarray(old.source_ids, dtype=np.int64), old.quant_bound)
                )

        copied: List[Tuple[int, np.ndarray, Optional[np.ndarray], float]] = []
        for page_id, rows, _ in replacements:
            if page_id in self._refresh_rows:
                values = self._refresh_rows[page_id]
            elif source_fn is not None:
                values = np.asarray(source_fn(rows.tolist()), dtype=np.float32)
            else:
                raise ValueError(f"no pinned source rows for page {page_id}")
            codes, scales, quant_bound = self._encode_rows(values)
            copied.append((page_id, codes, scales, quant_bound))

        published = 0
        with self._lock:
            for page_id, codes, scales, quant_bound in copied:
                old = self.pages.get(page_id)
                if (old is None or prepared_epoch != self.epoch or
                        self.refresh_version.get(page_id) != self.version[page_id]):
                    self.refresh_baseline.pop(page_id, None)
                    self.refresh_version.pop(page_id, None)
                    self._refresh_rows.pop(page_id, None)
                    continue
                baseline = self.refresh_baseline.pop(
                    page_id,
                    np.asarray(
                        [self.item_total.get(int(i), 0.0) for i in self.page_items[page_id]],
                        dtype=np.float64,
                    ),
                )
                self.refresh_version.pop(page_id, None)
                self._refresh_rows.pop(page_id, None)
                current = np.asarray(
                    [self.item_total.get(int(i), 0.0) for i in self.page_items[page_id]],
                    dtype=np.float64,
                )
                carried = np.maximum(current - baseline, 0.0).astype(np.float32)
                bound = float(carried.max()) if carried.size else 0.0
                still_dirty = bound > self.theta_max
                self.ledger[page_id] = carried
                self.version[page_id] += 1
                self.pages[page_id] = PackedPageView(
                    page_id=page_id,
                    source_ids=old.source_ids,
                    profiles=codes,
                    scales=scales,
                    drift_bound=bound,
                    quant_bound=quant_bound,
                    version=self.version[page_id],
                    dirty=still_dirty,
                    model_epoch=int(self.model_epoch),
                )
                if still_dirty:
                    self.dirty.add(page_id)
                else:
                    self.dirty.discard(page_id)
                published += 1
            return published

    def refresh(
        self,
        source_profiles: Optional[np.ndarray] = None,
        page_ids: Optional[Iterable[int]] = None,
        source_fn: Optional[Callable[[Sequence[int]], np.ndarray]] = None,
    ) -> int:
        selected, _ = self.prepare_refresh(page_ids)
        if not selected:
            return 0
        if source_profiles is not None:
            return self.publish_pages(source_profiles, selected)
        return self.publish_pages(page_ids=selected, source_fn=source_fn)

    def _snapshot_locked(
        self, source_ids: Sequence[int]
    ) -> Tuple[PackedPageView, ...]:
        ids = np.asarray(source_ids, dtype=np.int64)
        if self.resident_sids is None:
            return ()
        locations = np.searchsorted(self.resident_sids, ids)
        valid = (
            (locations < len(self.resident_sids))
            & (
                self.resident_sids[
                    np.minimum(locations, len(self.resident_sids) - 1)
                ]
                == ids
            )
        )
        page_ids = self.resident_page[locations[valid]]
        return tuple(self.pages[int(pid)] for pid in np.unique(page_ids))

    def snapshot(self, source_ids: Sequence[int]) -> Tuple[PackedPageView, ...]:
        with self._lock:
            return self._snapshot_locked(source_ids)

    def service(
        self,
        source_ids: Sequence[int],
        fallback_profiles: Optional[np.ndarray] = None,
        fallback_fn: Optional[Callable[[Sequence[int]], np.ndarray]] = None,
        strict_dirty: bool = True,
    ):
        ids = np.asarray(source_ids, dtype=np.int64)
        views = self.snapshot(ids)
        if strict_dirty and any(view.dirty for view in views):
            dirty = sorted(view.page_id for view in views if view.dirty)
            raise HotPageDirtyError(f"packed pages {dirty} require refresh")
        if (fallback_fn is None and fallback_profiles is None) and (
            len(views) < len(np.unique(ids)) or any(v.dirty for v in views)
        ):
            raise ValueError("a fallback is required for packed-page misses")

        by_id = {
            sid: (view, row)
            for view in views
            for row, sid in enumerate(view.source_ids)
        }
        out = np.empty((len(ids), self.dimension), dtype=np.float32)
        hit = 0
        missing: List[int] = []
        for i, sid in enumerate(ids.tolist()):
            entry = by_id.get(sid)
            if entry is None or entry[0].dirty:
                missing.append(i)
            else:
                view, row = entry
                out[i] = view.row(row)
                hit += 1

        if missing:
            missing_ids = ids[np.asarray(missing, dtype=np.int64)]
            if fallback_fn is not None:
                out[np.asarray(missing, dtype=np.int64)] = fallback_fn(missing_ids.tolist())
            else:
                out[np.asarray(missing, dtype=np.int64)] = np.asarray(
                    fallback_profiles, dtype=np.float32
                )[missing_ids]

        clean_views = tuple(view for view in views if not view.dirty)
        bound = max(
            (view.dispersion_bound for view in clean_views), default=0.0
        )
        return out, hit / max(1, len(ids)), bound

    def service_versioned(
        self,
        source_ids: Sequence[int],
        fallback_profiles: Optional[np.ndarray] = None,
        fallback_fn: Optional[Callable[[Sequence[int]], np.ndarray]] = None,
        strict_dirty: bool = True,
    ):
        """Atomically pin a model epoch together with immutable page views.

        The caller must use the returned epoch's embedding snapshot for query
        features and fallback recomputation.  Replaced pages can be reclaimed
        only after all readers holding their old epochs unpark; the Python
        reference implementation obtains that safety automatically through
        immutable views.
        """
        ids = np.asarray(source_ids, dtype=np.int64)
        with self._lock:
            pinned_epoch = int(self.model_epoch)
            views = self._snapshot_locked(ids)

        if strict_dirty and any(view.dirty for view in views):
            dirty = sorted(view.page_id for view in views if view.dirty)
            raise HotPageDirtyError(f"packed pages {dirty} require refresh")
        if (fallback_fn is None and fallback_profiles is None) and (
            len(views) < len(np.unique(ids)) or any(v.dirty for v in views)
        ):
            raise ValueError("a fallback is required for packed-page misses")

        by_id = {
            sid: (view, row)
            for view in views
            for row, sid in enumerate(view.source_ids)
        }
        out = np.empty((len(ids), self.dimension), dtype=np.float32)
        hit = 0
        missing: List[int] = []
        for i, sid in enumerate(ids.tolist()):
            entry = by_id.get(sid)
            if entry is None or entry[0].dirty:
                missing.append(i)
            else:
                view, row = entry
                out[i] = view.row(row)
                hit += 1

        if missing:
            missing_idx = np.asarray(missing, dtype=np.int64)
            missing_ids = ids[missing_idx]
            if fallback_fn is not None:
                out[missing_idx] = fallback_fn(missing_ids.tolist())
            else:
                out[missing_idx] = np.asarray(
                    fallback_profiles, dtype=np.float32
                )[missing_ids]

        clean_views = tuple(view for view in views if not view.dirty)
        bound = max(
            (view.dispersion_bound for view in clean_views), default=0.0
        )
        return out, hit / max(1, len(ids)), bound, pinned_epoch
