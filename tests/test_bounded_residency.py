#!/usr/bin/env python3
"""Tests for hard-capped packed pages and cold posting buckets."""
from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from systems.bounded_tier import build_bounded_tier  # noqa: E402
from systems.hot_pages import (BoundedPackedPageStore,  # noqa: E402
                               HotPageDirtyError)


def test_packed_pages_respect_hard_cap():
    sessions = [[i, i + 1, i + 2] for i in range(1000)]
    profiles = np.arange(4000, dtype=np.float32).reshape(1000, 4)
    store = BoundedPackedPageStore(
        30_000, dimension=4, page_size=16, theta_max=0.1)
    store.admit(range(1000), profiles, sessions)
    assert store.resident_bytes() <= 30_000
    assert len(store.pages) <= 63
    values, hit, bound = store.service([10, 11], profiles)
    assert hit == 1.0
    assert bound == 0.0
    np.testing.assert_array_equal(values, profiles[[10, 11]])


def test_hot_page_drift_during_refresh_remains_dirty():
    sessions = [[7] for _ in range(128)]
    profiles = np.zeros((128, 4), dtype=np.float32)
    store = BoundedPackedPageStore(100_000, 4, page_size=128,
                                   theta_max=0.1)
    store.admit(range(128), profiles, sessions)
    selected, _ = store.prepare_refresh([0])
    assert selected == (0,)
    store.update_drift({7: 0.2})
    store.publish_pages(profiles, selected)
    assert 0 in store.dirty_pages()
    try:
        store.service([10], profiles)
    except HotPageDirtyError:
        return
    raise AssertionError("drift during packed refresh was discarded")


def test_cold_posting_bucket_reads():
    root = Path(tempfile.mkdtemp(prefix="skc-tier-"))
    try:
        sessions = [[i, i + 1] for i in range(256)]
        profiles = np.zeros((256, 4), dtype=np.float32)
        store = build_bounded_tier(sessions, profiles, root,
                                   block_sessions=64, bands=8, rows=2,
                                   hot_blocks=0)
        store.memory_budget_bytes = 1
        ids, values = store.search(sessions[130], return_profiles=True)
        assert 130 in ids
        assert values.shape[1] == 4
        assert store.resident_bytes == 0
        assert store.disk_bucket_reads > 0
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    test_packed_pages_respect_hard_cap()
    test_hot_page_drift_during_refresh_remains_dirty()
    test_cold_posting_bucket_reads()
    print("bounded residency tests: ok")
