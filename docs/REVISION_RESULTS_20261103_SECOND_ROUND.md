# Second-round revision results — 2026-11-03

Archive:

`../revision_artifacts/cloud_results_20261103_revision2/icde_revision2_results.tar.gz`

SHA-256:

```text
61131e0ee83e25acc1107510416ab40ff6110535793c16bfb7d10b1509cb2132
```

## A3 actual-model drift

Each dataset trains MPLI for 30 epochs × 40 steps at batch size 256. The
saved operating checkpoints are:

- `results/real/diginetica_a3_cotrain_seed0.pt`
- `results/real/gowalla_a3_cotrain_seed0.pt`

Consecutive-epoch drift traces are stored under `results/systems/a3/`.

## F4 integrated audit

| Metric | Diginetica | Gowalla |
|---|---:|---:|
| packed/profile plane | 2.534 MB | 1.506 MB |
| peak posting plane | 6.932 MB | 7.110 MB |
| peak profile+posting total | 9.467 MB | 8.617 MB |
| Tier-2 projection audit | 12.401 MB | 5.482 MB |
| total with Tier-2 | 21.868 MB | 14.099 MB |
| posting recall@10 | 0.6803 | 0.5800 |
| profile checks / violations | 88,647 / 0 | 77,292 / 0 |
| Tier-2 checks / violations | 48,441 / 0 | 21,414 / 0 |
| exact-neighbor fresh HR@20 | 16.400% | 9.300% |
| posting fresh HR@20 | 16.433% | 9.133% |
| posting int8 cached HR@20 | 16.433% | 9.133% |

Process RSS was 1.55/1.62 GB, so cache-plane bytes must not be presented as
whole-process memory.

## Actual-model DAI-2 replay

At `theta=0.10`:

| Metric | Diginetica | Gowalla |
|---|---:|---:|
| intervals | 30 | 30 |
| pages | 1,226 | 2,314 |
| full refreshes | 36,780 | 69,420 |
| DAI-2 refreshes | 36,770 | 59,135 |
| saving | 0.03% | 14.82% |
| per-epoch refreshes | 36,770 | 53,291 |

## Held-out operating-checkpoint calibration

Adversarial directions are fitted on the first query split and evaluated on
the held-out split.

At profile dispersion 0.8:

- Diginetica baseline: 15.402% HR@20 over 7,603 held-out queries.
  - random: +0.03 pp;
  - correlated: +0.11 pp;
  - adversarial CE: -0.09 pp, CI [-0.20, +0.01];
  - margin attack: +0.14 pp.
- Gowalla baseline: 10.720% HR@20 over 10,000 held-out queries.
  - random: -0.01 pp;
  - correlated: -0.03 pp;
  - adversarial CE: -0.17 pp, CI [-0.31, -0.04];
  - margin attack: -0.06 pp.

The strong 5.46-point effect from the earlier isolated checkpoint does not
transfer when the attack direction is fitted on a separate split.

## Repeated SLA with precomputed neighbor ids

Each scenario uses 12 traces, 2,000 requests per trace.

### Real actual-model drift

| Dataset | Fresh p50/p99 | DAI-2 p50/p99 | Full p50/p99 |
|---|---:|---:|---:|
| Diginetica | 22.12 / 48.36 ms | 281.95 / 632.10 ms | 212.06 / 532.03 ms |
| Gowalla | 20.69 / 44.69 ms | 229.80 / 635.39 ms | 237.53 / 592.86 ms |

### Concentrated 80-item drift

| Dataset | Fresh p50/p99 | DAI-2 p50/p99 | Full p50/p99 |
|---|---:|---:|---:|
| Diginetica | 20.00 / 43.15 ms | 39.23 / 301.03 ms | 198.80 / 527.54 ms |
| Gowalla | 21.49 / 47.30 ms | 32.87 / 265.86 ms | 239.52 / 600.64 ms |

Fresh-on-demand dominates in both scenarios. DAI-2 is faster than full
invalidation only under concentrated drift.

## Posting-in-path audit

Each scenario uses ten 500-request traces and three arms. The fresh arm means
posting search followed by live-table profile recomputation.

| Dataset / scenario | Fresh p50/p99 | DAI-2 p50/p99 | Fresh QPS |
|---|---:|---:|---:|
| Diginetica real | 8.79 / 16.24 s | 18.21 / 33.86 s | 23.75 |
| Diginetica concentrated | 9.04 / 16.76 s | 13.37 / 24.39 s | 23.18 |
| Gowalla real | 16.66 / 28.83 s | 30.79 / 55.10 s | 14.83 |
| Gowalla concentrated | 16.92 / 29.43 s | 23.75 / 40.64 s | 14.57 |

The hard-cap posting tier is therefore not a 100-qps SLA configuration.
