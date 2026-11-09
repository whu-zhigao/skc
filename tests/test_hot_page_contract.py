#!/usr/bin/env python3
"""Contract tests for quantized packed pages and page-local updates."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from systems.hot_pages import BoundedPackedPageStore  # noqa: E402


def test_int8_page_enforces_reported_quantization_bound():
    sessions = [[i, i + 1] for i in range(256)]
    rng = np.random.default_rng(7)
    profiles = rng.normal(0, 0.08, size=(256, 64)).astype(np.float32)
    store = BoundedPackedPageStore(
        250_000, 64, page_size=128, theta_max=0.1,
        quantize=True, quant_bits=8,
    )
    store.admit(range(256), profiles, sessions)
    assert store.resident_bytes() <= 250_000
    values, hit, bound = store.service(range(256), profiles)
    assert hit == 1.0
    assert bound > 0.0
    actual = np.linalg.norm(values - profiles, axis=1)
    assert float(actual.max()) <= bound + 2e-6


def test_page_local_ledger_and_exact_refresh_carry():
    # Item 999 is shared by pages 0 and 1; refreshing page 0 must not reset 1.
    page = 128
    sessions = [[i % 20] + ([999] if i < 2 * page else [])
                for i in range(4 * page)]
    profiles = np.zeros((4 * page, 8), dtype=np.float32)
    store = BoundedPackedPageStore(
        1 << 20, 8, page_size=page, theta_max=0.1,
        quantize=False,
    )
    store.admit(range(4 * page), profiles, sessions)
    store.update_drift({999: 0.2})
    assert set(store.dirty_pages()) == {0, 1}
    selected, _ = store.prepare_refresh([0])
    assert selected == (0,)
    store.update_drift({999: 0.2})
    store.publish_pages(profiles, selected)
    # Drift arrived after preparation, so the replacement must remain dirty.
    assert 0 in store.dirty_pages()
    assert 1 in store.dirty_pages()


def test_refresh_pins_source_rows_at_preparation():
    sessions = [[7] for _ in range(128)]
    mutable_source = np.zeros((128, 4), dtype=np.float32)
    store = BoundedPackedPageStore(100_000, 4, page_size=128,
                                   theta_max=0.1)
    store.admit(range(128), mutable_source, sessions)
    selected, _ = store.prepare_refresh([0], source_profiles=mutable_source)
    # A later model-table mutation cannot tear the already pinned copy.
    mutable_source[:] = 1.0
    store.update_drift({7: 0.2})
    store.publish_pages(page_ids=selected)
    view = store.snapshot([10])[0]
    np.testing.assert_array_equal(view.profiles[0], np.zeros(4, dtype=np.float32))
    assert 0 in store.dirty_pages()


def test_service_version_pins_model_epoch_and_replacement_views():
    sessions = [[i, i + 1] for i in range(256)]
    profiles = np.zeros((256, 8), dtype=np.float32)
    store = BoundedPackedPageStore(
        1 << 20, 8, page_size=128, theta_max=0.1,
    )
    store.admit(range(256), profiles, sessions)
    before = store.snapshot(range(128))
    _, _, _, epoch0 = store.service_versioned(range(128), profiles)
    assert epoch0 == 0
    store.update_drift({0: 0.2})
    after = store.snapshot(range(128))
    _, _, _, epoch1 = store.service_versioned(
        range(128), profiles, strict_dirty=False
    )
    assert epoch1 == 1
    # The immutable pre-update snapshot remains alive for an existing reader,
    # while a new snapshot sees the dirty post-update metadata view.
    assert all(not view.dirty for view in before)
    assert all(view.dirty for view in after)


if __name__ == "__main__":
    test_int8_page_enforces_reported_quantization_bound()
    test_page_local_ledger_and_exact_refresh_carry()
    test_refresh_pins_source_rows_at_preparation()
    print("hot-page contract tests: ok")
