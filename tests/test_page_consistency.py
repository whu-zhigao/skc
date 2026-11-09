#!/usr/bin/env python3
"""Correctness tests for SKC consistency maintenance and cache misses."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from systems.page_consistency import (PageDirtyError,  # noqa: E402
                                      PageLocalDispersionStore)
from systems.skc import SKCConfig, SessionKVCache  # noqa: E402
from systems.igms import IGMS, IGMSConfig  # noqa: E402


def test_page_local_ledger_does_not_reset_other_pages() -> None:
    n, d, page = 4 * 32, 8, 32
    # Item 999 belongs to sessions on pages 0 and 1.
    sessions = [[i % 20] + ([999] if i < 2 * page else [])
                for i in range(n)]
    profiles = np.arange(n * d, dtype=np.float32).reshape(n, d)
    store = PageLocalDispersionStore(sessions, profiles, page_size=page,
                                      theta=0.1)
    store.update_drift({999: 0.2})
    assert set(store.dirty_pages()) == {0, 1}
    source = profiles.copy()
    store.refresh(source, page_ids=[0])
    assert 0 not in store.dirty_pages()
    assert 1 in store.dirty_pages()
    _, bound0 = store.service(list(range(page)))
    assert bound0 <= 0.1 + 1e-6
    try:
        store.service(list(range(page, 2 * page)))
    except PageDirtyError:
        pass
    else:  # pragma: no cover - the unrefreshed page must soft-miss
        raise AssertionError("page 1 was served after crossing theta")


def test_refresh_carries_drift_that_arrives_during_copy() -> None:
    n, page = 64, 32
    sessions = [[7] for _ in range(n)]
    profiles = np.zeros((n, 4), dtype=np.float32)
    store = PageLocalDispersionStore(sessions, profiles,
                                      page_size=page, theta=0.1)
    selected, _ = store.prepare_refresh([0], source_profiles=profiles)
    assert selected == (0,)
    # Drift arrives after preparation but before publication.
    store.update_drift({7: 0.2})
    store.publish_pages(profiles, selected)
    assert 0 in store.dirty_pages()
    try:
        store.service(list(range(page)))
    except PageDirtyError:
        return
    raise AssertionError("drift during refresh was discarded")


def test_concurrent_refresh_never_exposes_a_torn_page() -> None:
    n, d, page = 2048, 32, 128
    sessions = [[i % 100] for i in range(n)]
    profiles = np.zeros((n, d), dtype=np.float32)
    store = PageLocalDispersionStore(sessions, profiles, page_size=page,
                                      theta=0.1)
    stop = threading.Event()
    errors: list[Exception] = []
    observed = [0]

    def reader() -> None:
        rng = np.random.default_rng(17)
        try:
            while not stop.is_set():
                ids = rng.integers(0, n, size=320).tolist()
                try:
                    values, bound = store.service(ids)
                except PageDirtyError:
                    # Correct soft-miss behavior: never gather a dirty page.
                    observed[0] += 1
                    continue
                assert values.shape == (320, d)
                assert bound <= 0.1 + 1e-6
                observed[0] += 1
        except Exception as exc:  # pragma: no cover - surfaced by assertion
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for thread in threads:
        thread.start()
    source = profiles.copy()
    for version in range(1, 20):
        # Every replacement page is immutable and internally coherent.
        source[source.shape[0] // 2:] = float(version)
        store.update_drift({i: 0.2 for i in range(0, 100, 7)})
        store.refresh(source)
    stop.set()
    for thread in threads:
        thread.join()
    assert not errors, errors
    assert observed[0] > 0


def test_cache_miss_uses_source_and_nfa_scores_pages() -> None:
    n, d = 512, 8
    source = np.arange(n * d, dtype=np.float32).reshape(n, d)
    cfg = SKCConfig(page_size=128, memory_budget_mb=0.03, d=d,
                    admission="nfa")
    cache = SessionKVCache(cfg, source_profiles=source)
    ids = np.asarray([129, 130, 257])
    values, hit_rate = cache.fetch_profiles(ids)
    assert hit_rate == 0.0
    np.testing.assert_array_equal(values, source[ids])
    cache.observe_neighbors(range(256, 512))
    cache.admit(range(256, 512))
    # NFA must retain the high-frequency page and evict the low-frequency one.
    assert all(pid in cache.pages for pid in (2, 3))
    values, hit_rate = cache.fetch_profiles(np.asarray([257, 258]))
    assert hit_rate == 1.0
    np.testing.assert_array_equal(values, source[[257, 258]])


def test_igms_return_values_are_dequantized() -> None:
    items = [[1, 2, 3], [4, 5, 6]]
    profiles = np.asarray([[100.0, -50.0], [10.0, 20.0]],
                          dtype=np.float32)
    igms = IGMS(IGMSConfig(bands=2, sig_len=4, k_r=1,
                           bucket_layout="sorted", value_dim=2))
    for item, profile in zip(items, profiles):
        igms.insert(item, profile)
    ids, values = igms.search(items[0], k=1, return_values=True)
    np.testing.assert_allclose(values[0], profiles[0], rtol=1e-2,
                               atol=1e-2)


if __name__ == "__main__":
    test_page_local_ledger_does_not_reset_other_pages()
    test_refresh_carries_drift_that_arrives_during_copy()
    test_concurrent_refresh_never_exposes_a_torn_page()
    test_cache_miss_uses_source_and_nfa_scores_pages()
    test_igms_return_values_are_dequantized()
    print("page consistency tests: ok")
