# Third-round A800 revision results — 2026-11-06

Archive:

`../revision_artifacts/cloud_results_20261106_revision3/icde_revision3_gpu_results.tar.gz`

SHA-256:

```text
c883e20efe95797d74bd1d7c09a5f8701d80e35fbf9fbd042b434be9b274ba60
```

## GPU snapshot-publication matrix

Protocol:

- A800 CUDA default-stream optimizer updates;
- immutable GPU source snapshots;
- one reader CUDA stream per thread;
- atomic CPU metadata pinning;
- 1/2/4/8 readers;
- 20/80/320 dirty items;
- 160 publication cycles per case;
- 2 ms dirty window.

Aggregate:

| Metric | Value |
|---|---:|
| clean reads | 4,346 |
| soft misses | 52,906 |
| page publications | 71,226 |
| torn pages | 0 |
| version mismatches | 0 |
| acquisition p50 | 0.0014–0.0023 ms |
| publication p50 | 1.31–1.65 ms |

## A800 source-placement serving

The benchmark uses the trained 64-D A3 embedding table, real test sequences,
exact top-10 neighbor ids, Linux `pread`, `fadvise(DONTNEED)`, and the full
MPLI forward pass. The int8 cached arm is an offline active-set ceiling.

### Batch 32

| Dataset | storage fresh p50 | clean int8 p50 | reduction |
|---|---:|---:|---:|
| Diginetica | 131.789 ms | 105.205 ms | 20.2% |
| Gowalla | 109.098 ms | 90.193 ms | 17.3% |

### Batch 128

| Dataset | storage fresh p50 | clean int8 p50 | reduction |
|---|---:|---:|---:|
| Diginetica | 459.768 ms | 398.156 ms | 13.4% |
| Gowalla | 396.419 ms | 341.278 ms | 13.9% |

Materialization alone at batch 32 falls from 24.82/16.64 ms to
0.13/0.12 ms. Int8 preserves 99.87% top-20 logit overlap. The cache is
1.04/0.72 MB for 13,669/9,456 active sessions.

## Full-model projection drift

For each dataset, seed 1 co-trains all MPLI parameters for 30 epochs × 40
mini-batches.

| Dataset | epochs with projection change | mean projection-weight L2 | mean embedding-drift p99 |
|---|---:|---:|---:|
| Diginetica | 30/30 | 0.4284 | 0.1589 |
| Gowalla | 30/30 | 0.6407 | 0.1618 |

The global encoder's \(W_Q/W_K/W_V/W_O\) remain frozen in this recipe, while
`tare.w_qn`, `tare.w_n`, and intent query/key maps move. Changed projections
must therefore be invalidated; the embedding-only DAI-2 ledger does not cover
their changed view definition.

## Full-model DAI-2 replay at θ=0.10

| Dataset | Full refreshes | Per-epoch | Global item ledger | DAI-2 |
|---|---:|---:|---:|---:|
| Diginetica seed 1 | 36,780 | 36,747 | 36,780 | 36,748 |
| Gowalla seed 1 | 69,420 | 67,538 | 69,419 | 67,805 |

DAI-2 saves 0.09% on Diginetica and 2.33% on Gowalla. This reinforces the
paper's scoped claim: DAI-2 provides profile dispersion accounting, not a
universal throughput improvement or a projection-view solution.

## Verification

```text
python3 analysis/verify_paper_numbers.py
second-round paper headline numbers: verified
third-revision contract/source numbers: verified
```
