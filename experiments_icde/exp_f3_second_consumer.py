#!/usr/bin/env python3
"""F3: contract calibration on an independent cross-attention consumer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (RESULTS, load_data, load_model, neighbor_ids,  # noqa: E402
                    pool_profiles, serving_prefixes, write_json)
from experiments_icde.exp_f1_contract_calibration import (  # noqa: E402
    dispersion_stats, evaluate_profiles, perturb_embeddings)
from mplignn.data import pad_batch  # noqa: E402
from systems.independent_attention import (  # noqa: E402
    IndependentCrossAttentionConsumer)


@torch.no_grad()
def evaluate(model, sessions, neighbors, profiles, device, batch_size=64):
    hits = 0
    reciprocal = []
    for start in range(0, len(sessions), batch_size):
        chunk = sessions[start:start + batch_size]
        nb = neighbors[start:start + len(chunk)]
        profile_batch = torch.from_numpy(
            profiles[np.asarray(nb, dtype=np.int64)]).to(device)
        seqs, lens, target = pad_batch(chunk, str(device))
        logits = model(seqs, lens, profile_batch)
        ranks = (logits.argsort(dim=1, descending=True) ==
                 target[:, None]).nonzero(as_tuple=False)[:, 1] + 1
        hits += int((ranks <= 20).sum().item())
        reciprocal.extend((1.0 / ranks[ranks <= 20]).cpu().tolist())
    return {"HR@20": hits / len(sessions),
            "MRR@20": float(np.mean(reciprocal)) if reciprocal else 0.0}


def train_consumer(model, data, train_neighbors, profiles, device,
                   steps: int, batch_size: int):
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3)
    indices = np.arange(len(train_neighbors))
    rng = np.random.default_rng(7)
    for step in range(steps):
        chosen = rng.choice(indices, size=batch_size,
                            replace=len(indices) < batch_size)
        batch = [data.train[int(i)] for i in chosen]
        nb = train_neighbors[chosen]
        profile_batch = torch.from_numpy(
            profiles[np.asarray(nb, dtype=np.int64)]).to(device)
        seqs, lens, target = pad_batch(batch, str(device))
        logits = model(seqs, lens, profile_batch)
        loss = torch.nn.functional.cross_entropy(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 25 == 0:
            print(f"second-consumer step={step} loss={loss.item():.4f}",
                  flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base-ckpt", type=Path, default=None)
    parser.add_argument("--train-sessions", type=int, default=5000)
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--n-query", type=int, default=500)
    parser.add_argument("--drifts", default="0,0.10,0.20,0.40,0.80")
    parser.add_argument("--fractions", default="0,0.5,1.0")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.data)
    base = load_model(data, args.base_ckpt, device=str(device))
    embedding = base.embedding.weight.detach().cpu()
    model = IndependentCrossAttentionConsumer(embedding, embedding.shape[1])
    model.to(device)

    train_indices = np.linspace(0, len(data.train) - 1,
                                args.train_sessions, dtype=int)
    train_sessions = [data.train[int(i)] for i in train_indices]
    train_prefixes = serving_prefixes(train_sessions)
    train_nb = np.asarray(neighbor_ids(train_prefixes, data.train, 10),
                          dtype=np.int64)
    # Never use the training session itself as its neighbor: its full corpus
    # row contains the next-item target.
    train_nb = np.asarray([
        ([x for x in row if int(x) != int(train_indices[i])] +
         [row[-1]] * 10)[:10]
        for i, row in enumerate(train_nb)], dtype=np.int64)
    profiles = pool_profiles(embedding.numpy(), data.train)
    train_consumer(model, data, train_nb, profiles, device,
                   args.train_steps, 64)

    sessions = data.test[:min(args.n_query, len(data.test))]
    test_nb = np.asarray(neighbor_ids(serving_prefixes(sessions),
                                      data.train, 10), dtype=np.int64)
    active_ids = np.unique(test_nb.reshape(-1))
    points = []
    for drift in [float(x) for x in args.drifts.split(",")]:
        stale = pool_profiles(
            perturb_embeddings(embedding.numpy(), drift, 7), data.train)
        stats = dispersion_stats(profiles, stale, active_ids)
        for fraction in [float(x) for x in args.fractions.split(",")]:
            mixed = profiles.copy()
            count = int(round(fraction * test_nb.shape[1]))
            if count:
                choice = np.random.default_rng(71).permuted(
                    np.tile(np.arange(test_nb.shape[1]),
                            (len(test_nb), 1)), axis=1)[:, :count]
                rows, cols = np.meshgrid(np.arange(len(test_nb)),
                                         np.arange(count), indexing="ij")
                selected = test_nb[rows, choice]
                mixed[selected] = stale[selected]
            metrics = evaluate(model, sessions, test_nb, mixed, device)
            points.append({"item_drift": drift,
                           "stale_fraction": fraction,
                           **metrics, **stats})
            print(f"second drift={drift} f={fraction}: "
                  f"{metrics['HR@20']:.3f}", flush=True)

    result = {
        "dataset": args.data.stem,
        "consumer": "IndependentCrossAttentionConsumer",
        "device": str(device),
        "train_sessions": len(train_sessions),
        "train_steps": args.train_steps,
        "n_queries": len(sessions),
        "drifts": [float(x) for x in args.drifts.split(",")],
        "fractions": [float(x) for x in args.fractions.split(",")],
        "points": points,
    }
    write_json(args.out or (RESULTS / "f3" /
                            f"second_consumer_{args.data.stem}.json"),
               result)


if __name__ == "__main__":
    main()
