#!/usr/bin/env python3
"""D4: IGMS store layout microbenchmark (CPU-only, numpy). Builds
MinHash signatures + band postings with MinHashLSH from exp_d1, then
compares two posting-store layouts:
  (a) append-only log: (key32, sid) uint32 pairs appended into ONE
      preallocated buffer in 64KB-aligned blocks; per-band chains of
      block offsets form the band->offset index (block alignment is
      simulated exactly: a band grabs a fresh 64KB block whenever its
      current block is full);
  (b) dict-of-lists baseline (exp_d1's bucket structure).
N inserts then M random band queries (half hits on real keys, half
misses); reported: insert throughput (pairs/s), lookup P50/P99,
peak memory (tracemalloc; analytic fallback), write amplification
(bytes written / payload bytes)."""
import argparse
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import config_out, write_json
from exp_d1_ann_baselines import MinHashLSH

BLOCK_BYTES = 65536
PAIR_BYTES = 8                     # uint32 key32 + uint32 sid
PAIRS_PER_BLOCK = BLOCK_BYTES // PAIR_BYTES


class LogStore:
    """Layout (a): global append-only buffer, 64KB-aligned blocks,
    per-band chain of block offsets."""

    def __init__(self, n_pairs: int, n_bands: int):
        cap_blocks = int(np.ceil(n_pairs / PAIRS_PER_BLOCK)) + n_bands
        self.buf = np.empty(cap_blocks * PAIRS_PER_BLOCK * 2,
                            dtype=np.uint32)
        self.chains = [[] for _ in range(n_bands)]   # band->offset index
        self.cur = [-1] * n_bands                     # current block/band
        self.fill = np.zeros(n_bands, dtype=np.int64)
        self.n_blocks = 0
        self.bump = 0

    def _new_block(self, band: int):
        if (self.bump + 1) * PAIRS_PER_BLOCK * 2 > self.buf.size:
            self.buf = np.resize(self.buf, self.buf.size * 2)  # safety
        start = self.bump * PAIRS_PER_BLOCK * 2
        self.chains[band].append(start)
        self.cur[band] = start
        self.fill[band] = 0
        self.bump += 1
        self.n_blocks += 1

    def insert(self, band: int, key32: int, sid: int):
        if self.cur[band] < 0 or self.fill[band] >= PAIRS_PER_BLOCK:
            self._new_block(band)
        p = self.cur[band] + int(self.fill[band]) * 2
        self.buf[p] = key32
        self.buf[p + 1] = sid
        self.fill[band] += 1

    def lookup(self, band: int, key32: int):
        out = []
        for start in self.chains[band]:
            blk = self.buf[start:start + PAIRS_PER_BLOCK * 2]
            mask = blk[0::2] == np.uint32(key32)
            out.extend(blk[1::2][mask].tolist())
        return out

    def bytes_written(self) -> int:
        return self.n_blocks * BLOCK_BYTES

    def index_bytes(self) -> int:
        return sum(len(c) for c in self.chains) * 8 + len(self.chains) * 8


class DictStore:
    """Layout (b): dict-of-lists (exp_d1 bucket structure)."""

    def __init__(self, n_bands: int):
        self.buckets = [dict() for _ in range(n_bands)]

    def insert(self, band: int, key, sid: int):
        b = self.buckets[band]
        lst = b.get(key)
        if lst is None:
            b[key] = [sid]
        else:
            lst.append(sid)

    def lookup(self, band: int, key):
        return self.buckets[band].get(key, ())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", "--data", dest="dataset", type=Path,
                    required=True)
    ap.add_argument("--n-inserts", type=int, default=20000)
    ap.add_argument("--n-queries", type=int, default=2000)
    ap.add_argument("--r", type=int, default=2)
    ap.add_argument("--b", type=int, default=32,
                    help="bands per signature (default 32, the paper's "
                         "configuration; 2^5 x r=2)")
    ap.add_argument("--bands", default=None, metavar="LIST",
                    help="comma list of band counts to sweep, e.g. "
                         "8,16,32,64 (overrides --b; one per-config "
                         "JSON per value, tag encodes b)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="legacy single-file mode (always rewritten)")
    ap.add_argument("--force", action="store_true",
                    help="rerun even if the per-config JSON exists")
    args = ap.parse_args()

    if not args.dataset.exists():
        sys.exit(f"[d4] dataset not found: {args.dataset}\n"
                 f"       pass --dataset <repo>/code/data/diginetica.pkl")
    stem = args.dataset.stem
    band_list = ([int(x) for x in args.bands.split(",") if x.strip()]
                 if args.bands else [args.b])
    for b_val in band_list:
        out_path, skip = config_out(
            "d4", f"{stem}_n{args.n_inserts}_q{args.n_queries}"
                  f"_r{args.r}_b{b_val}_s{args.seed}",
            args.out if len(band_list) == 1 else None, args.force)
        if skip and not args.out:
            print(f"[d4] b={b_val}: exists, skipping: {out_path} "
                  "(--force to rerun)")
            continue
        from common import load_data
        rng = np.random.default_rng(args.seed)
        data = load_data(args.dataset)
        sessions = data.train[:args.n_inserts]
        n = len(sessions)
        print(f"[{stem}] building signatures for {n} sessions "
              f"(r={args.r}, b={b_val})...")
        lsh = MinHashLSH(sessions, r=args.r, b=b_val, seed=args.seed)
        keys = lsh.keys                   # per session: b band keys
        n_bands = b_val
        pairs = [(band, int(k) & 0xFFFFFFFF, int(k), sid)
                 for sid, key in enumerate(keys)
                 for band, k in enumerate(key)]
        payload_bytes = len(pairs) * PAIR_BYTES
        print(f"  {len(pairs)} postings "
              f"({payload_bytes / 1e6:.1f}MB payload)")

        # ---------------- layout (a): append-only log
        tracemalloc.start()
        t0 = time.perf_counter()
        log = LogStore(len(pairs), n_bands)
        for band, k32, _, sid in pairs:
            log.insert(band, k32, sid)
        t_ins_a = time.perf_counter() - t0
        _, peak_a = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # ---------------- layout (b): dict-of-lists
        tracemalloc.start()
        t0 = time.perf_counter()
        dct = DictStore(n_bands)
        for band, _, kfull, sid in pairs:
            dct.insert(band, kfull, sid)
        t_ins_b = time.perf_counter() - t0
        _, peak_b = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # ------------- queries: half hits on real keys, half misses
        hits = [pairs[i] for i in rng.choice(
            len(pairs), size=args.n_queries // 2, replace=False)]
        qs = [(band, k32, kfull) for band, k32, kfull, _ in hits]
        for _ in range(args.n_queries - len(qs)):
            qs.append((int(rng.integers(0, n_bands)),
                       int(rng.integers(0, 2 ** 32)), None))
        lat_a, lat_b, ok = [], [], 0
        for band, k32, kfull in qs:
            t0 = time.perf_counter()
            ra = log.lookup(band, k32)
            lat_a.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            rb = dct.lookup(band, kfull) if kfull is not None else ()
            lat_b.append(time.perf_counter() - t0)
            if kfull is not None:
                ok += int(sorted(ra) == sorted(rb))
        checked = sum(1 for q in qs if q[2] is not None)

        def pctl(v, q):
            return 1e6 * float(np.percentile(np.asarray(v), q))

        rows = {
            "log": {"insert_pairs_per_s": len(pairs) / t_ins_a,
                    "insert_s": t_ins_a,
                    "lookup_us": {"p50": pctl(lat_a, 50),
                                  "p99": pctl(lat_a, 99),
                                  "mean": 1e6 * float(np.mean(lat_a))},
                    "peak_mem_mb": peak_a / 1e6,
                    "buffer_mb": log.buf.nbytes / 1e6,
                    "bytes_written": log.bytes_written(),
                    "write_amplification":
                        log.bytes_written() / payload_bytes,
                    "index_bytes": log.index_bytes(),
                    "blocks": log.n_blocks},
            "dict": {"insert_pairs_per_s": len(pairs) / t_ins_b,
                     "insert_s": t_ins_b,
                     "lookup_us": {"p50": pctl(lat_b, 50),
                                   "p99": pctl(lat_b, 99),
                                   "mean": 1e6 * float(np.mean(lat_b))},
                     "peak_mem_mb": peak_b / 1e6,
                     "write_amplification": peak_b / payload_bytes}}
        for name, r in rows.items():
            print(f"  {name}: insert "
                  f"{r['insert_pairs_per_s'] / 1e3:.0f}K pairs/s, "
                  f"lookup p50/p99 "
                  f"{r['lookup_us']['p50']:.1f}/"
                  f"{r['lookup_us']['p99']:.1f}us, "
                  f"peak {r['peak_mem_mb']:.1f}MB, "
                  f"WA {r['write_amplification']:.3f}")
        print(f"  hit-path agreement (sorted sid lists): "
              f"{ok}/{checked}")

        md = ["| layout | insert Kpairs/s | lookup p50 us | lookup p99"
              " us | peak MB | write amp |",
              "|---|---|---|---|---|---|"]
        for name, r in rows.items():
            md.append(f"| {name} | "
                      f"{r['insert_pairs_per_s'] / 1e3:.0f} | "
                      f"{r['lookup_us']['p50']:.1f} | "
                      f"{r['lookup_us']['p99']:.1f} | "
                      f"{r['peak_mem_mb']:.1f} | "
                      f"{r['write_amplification']:.3f} |")
        print("\n".join(md))

        write_json(out_path, {
            "dataset": stem, "n_sessions": n,
            "n_postings": len(pairs),
            "r": args.r, "b": b_val, "block_bytes": BLOCK_BYTES,
            "pairs_per_block": PAIRS_PER_BLOCK,
            "bands_swept": band_list,
            "n_queries": len(qs), "hit_queries": checked,
            "hit_path_agreement": f"{ok}/{checked}",
            "payload_bytes": payload_bytes, "layouts": rows,
            "markdown_rows": md,
            "note": "Layout (a) simulates the IGMS append-only store: "
                    "one preallocated uint32 buffer, 64KB-aligned "
                    "blocks, band->offset chains as the index; write "
                    "amplification = allocated block bytes / payload "
                    "bytes (block utilization is the inverse). Layout "
                    "(b) is exp_d1's dict-of-lists; its write "
                    "amplification is estimated as tracemalloc peak / "
                    "payload (captures Python over-allocation). Lookup "
                    "latency is per (band,key) probe on this CPU node; "
                    "hits and misses are mixed 50/50, seeded."})


if __name__ == "__main__":
    main()
