#!/usr/bin/env python3
"""D3: concurrency. CPU: thread scaling {1,2,4,8,12,14} on the
profile-gather op. GPU: batch sweep {64..4096} QPS of gather+forward on
the A800. Forward stage depends on common.run_forward adaptation."""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (config_out, load_data, load_model, pool_profiles,
                    run_forward, write_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--out", default=None,
                    help="legacy single-file mode (always rewritten)")
    ap.add_argument("--force", action="store_true",
                    help="rerun even if the per-config JSON exists")
    args = ap.parse_args()

    stem = args.data.stem
    out_path, skip = config_out("d3", f"{stem}_sec{args.seconds:g}",
                                args.out, args.force)
    if skip:
        print(f"[d3] exists, skipping: {out_path} (--force to rerun)")
        return

    import torch
    data = load_data(args.data)
    model = load_model(data, args.ckpt, device="cpu")
    emb = model.embedding.weight.detach().numpy()
    P = torch.from_numpy(pool_profiles(emb, data.train))
    N = P.shape[0]
    rng = np.random.default_rng(0)

    cpu_rows = []
    for nthreads in (1, 2, 4, 8, 12, 14):
        stop = {"v": False}
        ids = torch.from_numpy(rng.integers(0, N, args.k))

        def worker():
            n = 0
            while not stop["v"]:
                P.index_select(0, ids)
                n += 1
            return n

        with ThreadPoolExecutor(nthreads) as ex:
            futs = [ex.submit(worker) for _ in range(nthreads)]
            time.sleep(args.seconds)
            stop["v"] = True
            total = sum(f.result() for f in futs)
        qps = total / args.seconds
        cpu_rows.append({"threads": nthreads, "qps": qps,
                         "scaling": qps / cpu_rows[0]["qps"]
                         if cpu_rows else 1.0})
        print(f"  {nthreads} threads: {qps:.0f} qps")

    gpu_rows = []
    if torch.cuda.is_available():
        dev = "cuda"
        model.to(dev)
        P_g = P.to(dev)
        for bs in (64, 256, 1024, 4096):
            ids = torch.randint(0, N, (bs * args.k,), device=dev)
            seq = torch.randint(0, data.n_items, (bs, 8), device=dev)
            prof = P_g[ids].view(bs, args.k, P.shape[1])
            try:
                for _ in range(5):
                    run_forward(model, seq, prof)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                nrep = 20
                for _ in range(nrep):
                    run_forward(model, seq, prof)
                torch.cuda.synchronize()
                gpu_rows.append({"batch": bs,
                                 "qps": bs * nrep
                                 / (time.perf_counter() - t0)})
            except RuntimeError as e:
                gpu_rows.append({"batch": bs, "skipped": str(e)})
                break

    write_json(out_path, {
        "dataset": stem, "cpu": cpu_rows, "gpu": gpu_rows,
        "note": "cpu op = torch index_select over the profile table; "
                "14 vCPU node cap respected"})


if __name__ == "__main__":
    main()
