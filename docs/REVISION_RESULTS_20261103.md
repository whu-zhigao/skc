# High-priority revision results — 2026-11-03

All numbers below were regenerated on the A800 node, retrieved through the
checksum-protected package, and verified locally by
`analysis/verify_paper_numbers.py`.

## Result provenance

| Evidence | Result path |
|---|---|
| Integrated hard-cap replay | `results/systems/f4/*_integrated_nq3000_*.json` |
| 61-interval drift traces | `results/systems/drift_trace_{diginetica,gowalla}.npz` |
| Exact page-local replay | `results/systems/dai2_pageledger_{diginetica,gowalla}.json` |
| High-power calibration | `results/systems/f5/*.json` |
| Repeated full-request SLA | `results/systems/f6/*_t30_*.json` |

Result archive:

`../revision_artifacts/cloud_results_20261103/icde_revision_core_results_20261103.tar.gz`

SHA-256:

```text
8c119db44f1895f481a1ab2a96a699348fe41e1ca6b042d6e109a1658532d865
```

## Integrated F4 audit

| Metric | Diginetica | Gowalla |
|---|---:|---:|
| corpus sessions | 156,898 | 296,121 |
| leakage-free queries | 3,000 | 3,000 |
| packed residency | 2.534 MB | 1.506 MB |
| posting residency | 6.927 MB | 6.749 MB |
| peak-temp-inclusive total | 9.463 MB | 8.256 MB |
| top-10 posting recall | 0.6803 | 0.5800 |
| packed-profile hit after refresh | 1.000 | 1.000 |
| dispersion checks | 88,647 | 77,292 |
| violations | 0 | 0 |

## 61-interval DAI-2 replay

At `theta=0.10`:

| Metric | Diginetica | Gowalla |
|---|---:|---:|
| intervals | 61 | 61 |
| pages | 1,226 | 2,314 |
| full-invalidation refreshes | 74,786 | 141,154 |
| DAI-2 refreshes | 72,739 | 138,197 |
| saving | 2.74% | 2.09% |
| membership | 1.63 MB | 1.71 MB |
| fp64 ledger | 3.25 MB | 3.42 MB |

## High-power calibration

Diginetica uses all 15,205 test queries and has 1,867 baseline HR@20 hits.
At profile dispersion 0.8:

- random perturbation: about `-0.16` percentage points;
- correlated perturbation: `-1.14` percentage points;
- task-gradient adversarial perturbation: `-5.46` percentage points,
  paired-bootstrap CI `[-5.82,-5.10]`.

Gowalla uses 10,000 queries but has only 33 baseline hits; it is reported for
transparency and is not used to license safety.

## Repeated 30-trace SLA

| Dataset | no-maintenance p99 | DAI-2 p99 | full-invalidation p99 |
|---|---:|---:|---:|
| Diginetica | 50.17 ms | 2,597.54 ms | 536.12 ms |
| Gowalla | 49.73 ms | 1,036.19 ms | 585.64 ms |

At the dense 500 ms maintenance cadence, DAI-2 is slower than full
invalidation. The paper therefore treats DAI-2 as an explicit safety/accounting
mechanism, not a universal throughput optimization.
