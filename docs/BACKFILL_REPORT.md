# BACKFILL_REPORT — 43 slots: 42 found, 1 missing (results root: /Volumes/workspace/VSworkspace/papers/Session Recommendation-WWW/code/results/systems)

map: `/Volumes/workspace/VSworkspace/papers/Session Recommendation-WWW/icde/paper/BACKFILL_MAP.md`

| slot | tex | status | found | value | n_files | sources |
|---|---|---|---|---|---|---|
| `tab:sharing:corpus_sessions` | L537 | SUITE | YES | 2.033e+05 ± 8.04e+04 (n=3) | 3 | a1/diginetica_nq3000_k10_w10.json, a1/diginetica_nq50_k10_w10.json +1 |
| `tab:sharing:unique_neighbors` | L537 | SUITE | YES | 1.116e+04 ± 1.01e+04 (n=3) | 3 | a1/diginetica_nq3000_k10_w10.json, a1/diginetica_nq50_k10_w10.json +1 |
| `tab:sharing:active_fraction` | L537 | SUITE | YES | 0.05855 ± 0.0651 (n=3) | 3 | a1/diginetica_nq3000_k10_w10.json, a1/diginetica_nq50_k10_w10.json +1 |
| `tab:sharing:oracle_hit_5mb` | L537 | SUITE | YES | 0.9926 ± 0.0147 (n=4) | 4 | b1/diginetica_nq3000_k10_d64.json, b1/diginetica_nq50_k10_d64.json +2 |
| `tab:gate:zero_rates` | L597 | SUITE | YES | -- | 2 | c2/diginetica_defaultckpt.json, c2/gowalla_mplignn_gowalla.json |
| `tab:gate:site_anchor` | L597 | SUITE | YES | 0.8237 | 2 | c2/diginetica_defaultckpt.json, c2/gowalla_mplignn_gowalla.json |
| `tab:gate:site_denoise` | L597 | SUITE | YES | 0 | 2 | c2/diginetica_defaultckpt.json, c2/gowalla_mplignn_gowalla.json |
| `tab:gate:tier3_latency` | L1012 | SUITE | YES | 2 non-scalar values (e.g. {"cached_gather": {"p50": 0.5277683540007274, "p99": 0.6189824928336023, "mean": 0.5255175820600562}, "dense_recompute_2) | 2 | c2/diginetica_defaultckpt.json, c2/gowalla_mplignn_gowalla.json |
| `tab:budgets:tier1_rows` | L623 | DERIVED | YES | 1.116e+04 ± 1.01e+04 (n=3) | 3 | a1/diginetica_nq3000_k10_w10.json, a1/diginetica_nq50_k10_w10.json +1 |
| `tab:budgets:tier2_rows` | L623 | DERIVED | YES | 3.396e+04 ± 6.05e+03 (n=2) | 2 | c1/diginetica.json, c1/gowalla.json |
| `tab:budgets:tier3_zero_rate` | L623 | DERIVED | YES | 1 ± 7.04e-06 (n=2) | 2 | c2/diginetica_defaultckpt.json, c2/gowalla_mplignn_gowalla.json |
| `cars_vs_greedy:hit` | L565/L1116 | SUITE | YES | 0.3834 ± 0.0788 (n=24) | 24 | b5/diginetica_20-40-40_cars_t29_nq1000_s0.json, b5/diginetica_20-40-40_cars_t29_nq1000_s1.json +22 |
| `nfa_greedy:hit` | L565 | SUITE | YES | 0.9297 ± 0.097 (n=8) | 8 | b5/diginetica_nfa-greedy_t29_nq1000_s0.json, b5/diginetica_nfa-greedy_t29_nq1000_s1.json +6 |
| `impl:insert_throughput` | L736 | SUITE | YES | 2.653e+05 ± 4.85e+04 (n=11) | 11 | d4/diginetica_n20000_q2000_r2_b64_s0.json, d4/diginetica_n20000_q2000_r2_b64_s1.json +9 |
| `tab:layouts:latency` | L828 | SUITE | YES | 6.146 ± 0.088 (n=2) | 2 | d4/diginetica_n400_q80_r2_b32_s0.json, d4/diginetica_n500_q100_r2_b32_s0.json |
| `tab:split:recall_int8` | L891 | SUITE | YES | 0.9811 ± 0.0126 (n=8) | 8 | b6/diginetica_int8-all_nq20_dr0.01_s0.json, b6/diginetica_int8-all_nq25_dr0.01_s0.json +6 |
| `tab:split:recall_int4` | L891 | SUITE | YES | 0.8779 ± 0.0243 (n=6) | 6 | b6/diginetica_int4-all_nq500_dr0.01_s0.json, b6/diginetica_int4-all_nq500_dr0.01_s1.json +4 |
| `tab:split:recall_int2` | L891 | SUITE | YES | 0.3495 ± 0.0405 (n=6) | 6 | b6/diginetica_int2-all_nq500_dr0.01_s0.json, b6/diginetica_int2-all_nq500_dr0.01_s1.json +4 |
| `tab:split:recall_utility_mixed` | L891 | SUITE | YES | 0.8217 ± 0.0615 (n=7) | 7 | b6/diginetica_utility-mixed_nq20_dr0.01_s0.json, b6/diginetica_utility-mixed_nq500_dr0.01_s0.json +5 |
| `tab:split:recall_elbow_mixed` | L891 | SUITE | YES | 0.5634 ± 0.202 (n=6) | 6 | b6/diginetica_elbow-mixed_nq500_dr0.01_s0.json, b6/diginetica_elbow-mixed_nq500_dr0.01_s1.json +4 |
| `tab:split:recall_cost_aware` | L891 | SUITE | YES | 0.08132 ± 0.0365 (n=24) | 24 | b5/diginetica_20-40-40_cars_t29_nq1000_s0.json, b5/diginetica_20-40-40_cars_t29_nq1000_s1.json +22 |
| `tab:split:hit_greedy` | L891 | SUITE | YES | 0.9297 ± 0.097 (n=8) | 8 | b5/diginetica_nfa-greedy_t29_nq1000_s0.json, b5/diginetica_nfa-greedy_t29_nq1000_s1.json +6 |
| `tab:split:p99_greedy` | L891 | SUITE | YES | 0.216 ± 0.0919 (n=8) | 8 | b5/diginetica_nfa-greedy_t29_nq1000_s0.json, b5/diginetica_nfa-greedy_t29_nq1000_s1.json +6 |
| `tab:split:hit_bytematched` | L891 | SUITE | YES | 1 ± 0 (n=2) | 2 | b5/diginetica_bm-int4_nfa_t5_nq40_s0.json, b5/diginetica_bm-int8_nfa_t5_nq40_s0.json |
| `tab:split:p99_bytematched` | L891 | SUITE | YES | 0.09338 ± 0.0203 (n=2) | 2 | b5/diginetica_bm-int4_nfa_t5_nq40_s0.json, b5/diginetica_bm-int8_nfa_t5_nq40_s0.json |
| `b6:e2e_hr` | L891 | SUITE | YES | 0.52 | 33 | b6/diginetica_elbow-mixed_nq500_dr0.01_s0.json, b6/diginetica_elbow-mixed_nq500_dr0.01_s1.json +31 |
| `e2e:b2_latencies` | L911 | SUITE | YES | -- | 1 | b2/diginetica_nq20_bsmax64.json |
| `e2e:d3_cpu_scaling` | L1034 | SUITE | YES | series len=6 (e.g. [{"threads": 1, "qps": 381464.0, "scaling": 1.0}, {"threads": 2, "qps": 368799.0, "scaling": 0.9667989639913596}, {"thre) | 3 | d3/diginetica_sec1.json, d3/diginetica_sec3.json +1 |
| `e2e:d3_gpu_qps` | L1034 | SUITE | YES | series len=0 (e.g. []) | 3 | d3/diginetica_sec1.json, d3/diginetica_sec3.json +1 |
| `cotrain:intermediate_R` | L957 | SUITE | YES | series len=3 (e.g. [{"epoch": 1, "recall_at_20": 0.9562, "n_refreshes": 6, "refresh_s": 0.2605, "mean_loss": 6.760905847549439}, {"epoch": ) | 27 | b7/diginetica_train_R16_e3_nq500_s0.json, b7/diginetica_train_R16_e3_nq500_s1.json +25 |
| `dai:skip_rates` | L992-999 | SUITE | YES | series len=11 (e.g. [{"theta": 0.001, "mean_skip_rate": 0.0}, {"theta": 0.003973488602787257, "mean_skip_rate": 0.0}, {"theta": 0.0051679924) | 2 | b3/diginetica_p128_live.json, b3/gowalla_p128_live.json |
| `dai:drift_quantiles` | L999 | SUITE | YES | 6 non-scalar values (e.g. {"q10": 0.003878779709339142, "q25": 0.006664014887064695, "q50": 0.042461223900318146, "q75": 0.0745057687163353, "q90") | 6 | a2/diginetica_inline_steps2000_s0.json, a2/diginetica_inline_steps2000_s1.json +4 |
| `dai:epoch_savings` | L999 | SUITE | YES | 0.9827 ± 0.0353 (n=2) | 2 | b3/diginetica_p128_live.json, b3/gowalla_p128_live.json |
| `ann:minhash_vs_faiss` | L1021 | SUITE | YES | 0.5289 ± 0.0435 (n=4) | 4 | d1/diginetica_nq30_k10_minhash_lsh.json, d1/diginetica_nq40_k10_minhash_lsh.json +2 |
| `ann:faiss` | L1021 | SUITE | YES | 4 non-scalar values (e.g. "skipped: pip install faiss-cpu") | 4 | d1/diginetica_nq30_k10_faiss.json, d1/diginetica_nq40_k10_faiss.json +2 |
| `ann:yoochoose_scaling` | L1021 | SUITE | YES | 0.154 | 1 | d2/yoochoose_quarter_ann_nq50_k10_cap5000_minhash_lsh.json |
| `ann:yoochoose_pareto` | L1021 | SUITE | YES | series len=7 (e.g. [{"budget_mb": 1, "budget_sessions": 3906, "oracle": 1.0, "nfa_offline": 1.0, "lru": 0.168, "lfu": 0.0, "arc": 0.168, "t) | 1 | d2/yoochoose_quarter_pareto_nq50_k10_d64_s0.json |
| `b7:reference_row` | L957/L982 | SUITE | YES | 0.9225 ± 0 (n=2) | 34 | b7/diginetica_mix_r0_nq10_s0.json, b7/diginetica_mix_r0_nq12_s0.json +32 |
| `ablation:cotrain_tables` | L930/L962 | COMPANION-SOURCED | no | -- | 0 |  |
| `fig:workload` | L816 | SUITE | YES | -- | 3 | a1/diginetica_nq3000_k10_w10.json, a1/diginetica_nq50_k10_w10.json +1 |
| `fig:pareto` | L867 | SUITE | YES | -- | 4 | b1/diginetica_nq3000_k10_d64.json, b1/diginetica_nq50_k10_d64.json +2 |
| `fig:mixed` | L982 | SUITE | YES | -- | 3 | b4/diginetica_nq1000_mixedgen.json, b4/diginetica_nq20_mixedgen.json +1 |
| `b7:mix_grid` | L982 tail | SUITE | YES | 0.9821 ± 0.0149 (n=34) | 34 | b7/diginetica_mix_r0_nq10_s0.json, b7/diginetica_mix_r0_nq12_s0.json +32 |
