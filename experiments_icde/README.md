# ICDE 实验套件（A1-D3）

安装位置：`icde/code/experiments_icde/`（脚本自动定位仓库 `code/` 根，
即含 `mplignn/` 的目录；也可 `export SKC_CODE_ROOT=/path/to/repo/code`）。
产物写入 `code/results/systems/<exp>/<dataset>_<关键参数>_<seed>.json`
（一配置一文件；已存在自动跳过，`--force` 重跑；`--out` 保持旧单文件
模式）——论文所有数字必须能从中再生。
硬件：A800-80GB ×1，14 vCPU Xeon Gold 6348，120GB RAM。

## 运行
    cd <repo>/code
    DATA=data/diginetica.pkl bash ../icde/code/experiments_icde/run_all.sh
    # 并行全矩阵（GPU 节点，详见 GPU_GUIDE.md）：
    bash ../icde/code/experiments_icde/run_all_parallel.sh --smoke

依赖：torch, numpy；D1 可选 `pip install faiss-cpu`。

## 顺序、依赖与论文槽位对照（合并版 main_icde.tex）

| 顺序 | 脚本 | 依赖 | 论文位置 |
|---|---|---|---|
| 1 | exp_a1_workload.py | — | §V-B workload（hub 稳定性 Fig.） |
| 2 | exp_a2_drift_trace.py | — | §V-g DAI（真实漂移分布） |
| 3 | exp_b1_cache_pareto.py | — | §V-d Tier-1 Pareto Fig. |
| 4 | exp_b2_e2e_latency.py | run_forward 适配 | §V-e E2E 延迟 |
| 5 | exp_b3_dai_replay.py | A2 的 npz | §V-g DAI 加速 |
| 6 | exp_b4_mixed_generation.py | 两个世代 ckpt（OLD_CKPT） | §V-f/g 一致性边界 Fig. |
| 7 | exp_c1_tier2.py | A2 npz（可选） | §V Tier-2 |
| 8 | exp_c2_gate_spectrum.py | — | §V-h 门谱系复测 |
| 9 | exp_d1_ann_baselines.py | faiss 可缺省 | §V-i ANN 基线 |
| 10 | exp_d3_concurrency.py | — | §V-j 并发/GPU 吞吐 |
| 11 | exp_b5_budget_split.py | — | §V-d 三层预算切分（NFA 贪心 vs 固定切分） |
| 12 | exp_b6_precision.py | 一个 ckpt | §V-d Tier-1 精度谱系 + 漂移重标定 |
| 13 | exp_b7_staleness.py | 两个世代 ckpt（OLD_CKPT） | §V-g 陈旧度鲁棒（可选 --train 刷新节奏） |
| 14 | exp_d4_igms_layout.py | — | §V-i IGMS 追加日志 vs dict 布局微基准 |
| 15 | run_parallel.py | — | 并行驱动器：CPU 池 + GPU 槽，幂等续跑，manifest |
| 16 | GPU_GUIDE.md | — | A800 节点运行/监控/断点续跑/论文回填指南 |
| 17 | check_env.py | — | 上节点预检：python/numpy/torch、CODE_ROOT、数据与 ckpt 逐项 FOUND/MISSING（`SKC_DATA_ROOT` 可重定向数据目录） |
| 18 | backfill_report.py | — | 论文槽位覆盖报告：按 `paper/BACKFILL_MAP.md` 的机读映射扫描 results/systems，逐槽位 FOUND/MISSING + 取值 + 多种子聚合，产出 BACKFILL_REPORT.{md,json} |
| 19 | exp_d2_yoochoose.py | yoochoose_quarter.pkl + mplignn_yoochoose_quarter.pt（自动回退链） | §V-i Yoochoose-1/4 扩展性（workload + Pareto + ANN 三段，`--sections` 可选段；ANN 段 `--ann-corpus-cap` 子采样并记录在 JSON） |

复审扩展（v3，逐项见脚本 `--help` 与 BACKFILL_MAP 的 schema notes）：

- **准入策略族**（`common.py`，引用见 `ADMISSION_CITATIONS`）：
  tinylfu（Einziger et al., SCN'17）、w-tinylfu（Einziger et al., TOS'18）、
  lecar（Shi et al., HotNets'18）、greedydualsize（Cao & Irani, SOSP'97）。
  `b1 --admission all|lru,lfu,arc,...`（默认全族）与
  `b5 --admission {nfa,cars,uniform,lru,arc,tinylfu,wtinylfu,lecar,greedydualsize}`
  （在线策略按到达序回放后物化驻留集）；b1 另有 `exact_reference` 全缓存上界点。
- `b7 --arm {cached,nocache}`（nocache=每批从活表重算邻居档案，LIGHT-GRADE）
  与 `--ref-seeds 0,1,2,3,4`（会话内参考行 mean±std，写入每个 b7 JSON）。
- `b6 --e2e`：反量化档案过真实模型，报 HR@20/MRR@10（含 fp32 参考；文件需不存在或 `--force`）。
- `d1`/`d2` 精确参考行（`*_exact.json`：全扫描精确 Jaccard 上界 + 计时）。
- `b5 --byte-matched [--match-bits 8,4,2]`：字节对齐的精度×命中率/P99 对比（`*_bm-int*` 文件）。
- `d4 --bands 8,16,32,64`（默认 b=32，每带宽一文件）；`b3` θ 行带 `theta_label`（q10..q99/abs）。
- `c2` 新增 `gate_sites`：anchor/SAtt 与 denoise/NAtt 两个 gate 位点的真实
  entmax 精确零率（真实 ckpt 批次、模型自带 `_sparse_attn`）。

## 上 GPU 节点前
0. 先跑 `python3 check_env.py`（必须 READY；数据集不在 icde/code 内，
   见 GPU_GUIDE (c) 的三种提供方式与文件清单）。`common.py::run_forward()`
   已按 `mplignn/model.py` 真实签名 `forward(seqs, lens, a, a_r, r0)` 接线。
1. B4/B7 需要旧世代 ckpt：`export OLD_CKPT=...abl_no_anchor_{ds}.pt`
   （不设时驱动器按数据集自动回退到 `results/real/` 里的默认对）。

## 诚信红线
- A2 的漂移是真实训练产物，替换旧 bench 的 randn 模拟；论文中 DAI 的
  51.5%/63.6% 两处数字在真实轨迹复测后必须替换。
- B1 按到达序回放；旧 bench 的 NFA/LRU 是离线 top-K，只能当天花板引用。
- 任何数字不得手写进 tex，全部走 JSON → tex 管线。
