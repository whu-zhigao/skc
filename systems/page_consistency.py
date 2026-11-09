"""Bounded-dispersion page maintenance for SKC.

This module is the non-budgeted reference implementation of the consistency
substrate.  As in the packed-page store, replacement rows are copied outside
the metadata lock and published atomically.  A CSR item index makes drift
updates proportional to affected item/page pairs rather than to all pages.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .source_snapshot import SourceSnapshot, as_snapshot


class PageDirtyError(RuntimeError):
    """A requested page crossed its drift threshold and needs refresh."""


@dataclass(frozen=True)
class PageView:
    page_id: int
    session_ids: Tuple[int, ...]
    profiles: np.ndarray
    drift_bound: float
    quant_bound: float
    version: int
    dirty: bool = False

    def __post_init__(self) -> None:
        if self.profiles.ndim != 2:
            raise ValueError("PageView.profiles must be a 2-D array")
        if len(self.session_ids) != len(self.profiles):
            raise ValueError("page ids and profiles have different lengths")
        self.profiles.flags.writeable = False

    @property
    def dispersion_bound(self) -> float:
        return float(self.drift_bound + self.quant_bound)


class PageLocalDispersionStore:
    """Immutable natural pages with page-local cumulative-drift bookkeeping."""

    def __init__(
        self,
        session_items: Sequence[Iterable[int]],
        profiles: np.ndarray,
        page_size: int = 128,
        theta: float = 0.1,
        quant_bound: float = 0.0,
    ) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        profiles = np.asarray(profiles, dtype=np.float32)
        if profiles.ndim != 2 or len(profiles) != len(session_items):
            raise ValueError("profiles must have one row per session")

        self.page_size = int(page_size)
        self.theta = float(theta)
        self.quant_bound = float(quant_bound)
        self._lock = threading.RLock()
        self._pages: Dict[int, PageView] = {}
        self._page_items: Dict[int, np.ndarray] = {}
        self._ledger: Dict[int, np.ndarray] = {}
        self._version: Dict[int, int] = {}
        self._dirty: set[int] = set()
        self.publication_epoch = 0
        self._refresh_baseline: Dict[int, np.ndarray] = {}
        self._refresh_version: Dict[int, int] = {}
        self._refresh_rows: Dict[int, np.ndarray] = {}

        max_item = -1
        for items in session_items:
            for item in items:
                max_item = max(max_item, int(item))
        self.item_total = np.zeros((max_item + 1,), dtype=np.float64)

        pairs: List[Tuple[int, int, int]] = []
        for start in range(0, len(session_items), self.page_size):
            pid = start // self.page_size
            ids = tuple(range(start, min(start + self.page_size, len(session_items))))
            items = sorted(
                {
                    int(i)
                    for session in session_items[start : start + self.page_size]
                    for i in session
                }
            )
            self._page_items[pid] = np.asarray(items, dtype=np.int32)
            self._ledger[pid] = np.zeros(len(items), dtype=np.float64)
            self._version[pid] = 0
            self._pages[pid] = PageView(
                page_id=pid,
                session_ids=ids,
                profiles=profiles[list(ids)].copy(),
                drift_bound=0.0,
                quant_bound=self.quant_bound,
                version=0,
                dirty=False,
            )
            for position, item in enumerate(items):
                pairs.append((item, pid, position))

        pairs.sort(key=lambda x: x[0])
        pair_items = np.fromiter((x[0] for x in pairs), dtype=np.int32, count=len(pairs))
        self._item_values = np.unique(pair_items)
        rows = np.searchsorted(self._item_values, pair_items)
        counts = np.bincount(rows, minlength=len(self._item_values))
        self._item_offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=self._item_offsets[1:])
        self._item_pages = np.fromiter((x[1] for x in pairs), dtype=np.int32, count=len(pairs))
        self._item_positions = np.fromiter((x[2] for x in pairs), dtype=np.int32, count=len(pairs))

    @property
    def n_pages(self) -> int:
        return len(self._pages)

    def ledger_bytes(self) -> int:
        return sum(arr.nbytes for arr in self._ledger.values())

    def item_index_bytes(self) -> int:
        return int(
            self._item_values.nbytes
            + self._item_offsets.nbytes
            + self._item_pages.nbytes
            + self._item_positions.nbytes
        )

    def _affected(self, item: int) -> Iterable[Tuple[int, int]]:
        row = int(np.searchsorted(self._item_values, item))
        if row >= len(self._item_values) or int(self._item_values[row]) != item:
            return ()
        begin, end = int(self._item_offsets[row]), int(self._item_offsets[row + 1])
        return (
            (int(self._item_pages[j]), int(self._item_positions[j]))
            for j in range(begin, end)
        )

    def update_drift(self, item_drift: Mapping[int, float]) -> None:
        if not item_drift:
            return
        item_ids = np.fromiter((int(i) for i in item_drift), dtype=np.int64, count=len(item_drift))
        values = np.fromiter((float(v) for v in item_drift.values()), dtype=np.float64, count=len(item_drift))
        needed = max(int(item_ids.max(initial=-1)) + 1, len(self.item_total))
        if needed > len(self.item_total):
            grown = np.zeros(needed, dtype=np.float64)
            grown[: len(self.item_total)] = self.item_total
            self.item_total = grown

        with self._lock:
            self.publication_epoch += 1
            self.item_total[item_ids] += values
            touched = set()
            for item, value in item_drift.items():
                for pid, position in self._affected(int(item)):
                    self._ledger[pid][position] += float(value)
                    touched.add(pid)
            for pid in touched:
                old = self._pages[pid]
                bound = float(self._ledger[pid].max())
                dirty = bound > self.theta
                if dirty:
                    self._dirty.add(pid)
                else:
                    self._dirty.discard(pid)
                if old.drift_bound != bound or old.dirty != dirty:
                    self._pages[pid] = PageView(
                        page_id=old.page_id,
                        session_ids=old.session_ids,
                        profiles=old.profiles,
                        drift_bound=bound,
                        quant_bound=old.quant_bound,
                        version=old.version,
                        dirty=dirty,
                    )

    def dirty_pages(self) -> Tuple[int, ...]:
        with self._lock:
            return tuple(sorted(self._dirty))

    def prepare_refresh(
        self,
        page_ids: Optional[Iterable[int]] = None,
        source_profiles: Optional[SourceSnapshot | np.ndarray] = None,
    ) -> Tuple[Tuple[int, ...], int]:
        selected_tuple = tuple(sorted(set(page_ids) if page_ids is not None else self._dirty))
        snapshot = as_snapshot(source_profiles) if source_profiles is not None else None
        with self._lock:
            for pid in selected_tuple:
                if pid not in self._page_items:
                    raise KeyError(f"unknown page {pid}")
                self._refresh_baseline[pid] = self.item_total[self._page_items[pid]].copy()
                self._refresh_version[pid] = self._version[pid]
                if snapshot is not None:
                    rows = np.asarray(self._pages[pid].session_ids, dtype=np.int64)
                    copied = snapshot.profiles[rows].copy()
                    copied.flags.writeable = False
                    self._refresh_rows[pid] = copied
            return selected_tuple, int(len(self.item_total))

    def publish_pages(
        self,
        source_profiles: Optional[SourceSnapshot | np.ndarray] = None,
        page_ids: Optional[Iterable[int]] = None,
    ) -> int:
        selected_tuple = tuple(sorted(set(page_ids) if page_ids is not None else self._refresh_baseline))
        if source_profiles is not None:
            snapshot = as_snapshot(source_profiles)
            if set(self._refresh_rows) - set(selected_tuple):
                # Refresh always pins selected rows at preparation time.
                self._refresh_rows.clear()
            for pid in selected_tuple:
                if pid not in self._refresh_rows:
                    rows = np.asarray(self._pages[pid].session_ids, dtype=np.int64)
                    copied = snapshot.profiles[rows].copy()
                    copied.flags.writeable = False
                    self._refresh_rows[pid] = copied

        if not self._refresh_rows:
            raise ValueError("prepare_refresh must pin an immutable source snapshot")

        # The source rows were copied from a read-only snapshot at preparation.
        # Encoding/publication work outside the metadata lock cannot observe a
        # later model update.
        snapshots: List[Tuple[int, int, np.ndarray]] = []
        with self._lock:
            for pid in selected_tuple:
                if pid not in self._pages:
                    raise KeyError(f"unknown page {pid}")
                old = self._pages[pid]
                snapshots.append((pid, self._version[pid], np.asarray(old.session_ids, dtype=np.int64)))
        copies = [
            (pid, version, self._refresh_rows[pid])
            for pid, version, rows in snapshots
        ]

        published = 0
        with self._lock:
            for pid, prepared_version, rows in copies:
                old = self._pages.get(pid)
                if old is None or self._version[pid] != prepared_version:
                    self._refresh_baseline.pop(pid, None)
                    self._refresh_version.pop(pid, None)
                    self._refresh_rows.pop(pid, None)
                    continue
                baseline = self._refresh_baseline.pop(
                    pid, self.item_total[self._page_items[pid]].copy()
                )
                self._refresh_version.pop(pid, None)
                self._refresh_rows.pop(pid, None)
                carried = np.maximum(
                    self.item_total[self._page_items[pid]] - baseline, 0.0
                ).astype(np.float32)
                bound = float(carried.max()) if carried.size else 0.0
                still_dirty = bound > self.theta
                self._ledger[pid] = carried
                self._version[pid] += 1
                self._pages[pid] = PageView(
                    page_id=pid,
                    session_ids=old.session_ids,
                    profiles=rows,
                    drift_bound=bound,
                    quant_bound=self.quant_bound,
                    version=self._version[pid],
                    dirty=still_dirty,
                )
                if still_dirty:
                    self._dirty.add(pid)
                else:
                    self._dirty.discard(pid)
                published += 1
        return published

    def refresh(
        self,
        source_profiles: SourceSnapshot | np.ndarray,
        page_ids: Optional[Iterable[int]] = None,
    ) -> int:
        selected, _ = self.prepare_refresh(page_ids, source_profiles=source_profiles)
        if not selected:
            return 0
        return self.publish_pages(source_profiles, selected)

    def snapshot(self, session_ids: Sequence[int]) -> Tuple[PageView, ...]:
        with self._lock:
            by_page: Dict[int, List[int]] = {}
            for sid in session_ids:
                pid = int(sid) // self.page_size
                by_page.setdefault(pid, []).append(int(sid))
            return tuple(self._pages[pid] for pid in by_page)

    def snapshot_versioned(
        self, session_ids: Sequence[int]
    ) -> Tuple[Tuple[PageView, ...], int]:
        """Atomically return immutable page views and the model publication epoch."""
        with self._lock:
            return self.snapshot(session_ids), int(self.publication_epoch)

    def service_versioned(
        self, session_ids: Sequence[int]
    ) -> Tuple[np.ndarray, float, int]:
        views, epoch = self.snapshot_versioned(session_ids)
        if any(view.dirty for view in views):
            dirty = sorted(view.page_id for view in views if view.dirty)
            raise PageDirtyError(f"pages {dirty} crossed the dispersion bound")
        by_id = {
            sid: (view, row)
            for view in views
            for row, sid in enumerate(view.session_ids)
        }
        out = np.stack([by_id[int(sid)][0].profiles[by_id[int(sid)][1]] for sid in session_ids])
        bound = max((view.dispersion_bound for view in views), default=0.0)
        return out, float(bound), epoch

    def service(self, session_ids: Sequence[int]) -> Tuple[np.ndarray, float]:
        ids = np.asarray(session_ids, dtype=np.int64)
        views = self.snapshot(session_ids)
        if any(view.dirty for view in views):
            dirty = sorted(view.page_id for view in views if view.dirty)
            raise PageDirtyError(f"pages {dirty} crossed the dispersion bound")
        by_id = {
            sid: (view, row)
            for view in views
            for row, sid in enumerate(view.session_ids)
        }
        out = np.stack([by_id[int(sid)][0].profiles[by_id[int(sid)][1]] for sid in ids.tolist()])
        bound = max((view.dispersion_bound for view in views), default=0.0)
        return out, bound
