"""Second-consumer contract replication (protocol-correct): trains the
consumer model with neighbor profiles served from a Tier-1 cache whose
refresh cadence R varies; endpoint parity R=1 vs R=inf on two consumer
configurations per dataset."""
import sys, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path('/Volumes/workspace/VSworkspace/papers/Session Recommendation-WWW/icde/code')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'experiments_icde'))
from common import load_data, RESULTS, run_forward, jaccard_topk
from mplignn.model import MPLI_GNN

def load_consumer(data, ckpt):
    flags = dict(no_residual='no_residual' in ckpt, no_anchor='no_anchor' in ckpt,
                 no_denoise='no_denoise' in ckpt)
    m = MPLI_GNN(n_items=data.n_items, d=64, local_layers=2, n_heads=2,
                 global_blocks=2, vocab_blocks=1, k_r=10, **flags)
    m.load_state_dict(torch.load(ROOT/'results/real'/ckpt, map_location='cpu', weights_only=True))
    return m
def norm(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
def top20(Q, P):
    out = []
    for i in range(0, len(Q), 64):
        out.append(np.argpartition(-(Q[i:i+64] @ P.T), 20, axis=1)[:, :20])
    return np.concatenate(out)
def pool(emb, sess): return np.stack([emb[np.asarray(sorted(set(s)))].mean(axis=0) for s in sess])

CONSUMERS = {
  'diginetica': [('default_no_residual', 'abl_no_residual_diginetica.pt'),
                 ('no_anchor', 'abl_no_anchor_diginetica.pt')],
  'gowalla':    [('full', 'mplignn_gowalla.pt'),
                 ('no_denoise', 'abl_nodenoise_gowalla.pt')],
}
EPOCHS, STEPS, BS, NQ, K = 1, 30, 256, 200, 10
results = {}; t00 = time.perf_counter()
for ds, cons in CONSUMERS.items():
    data = load_data(ROOT/'data'/f'{ds}.pkl')
    sessions = [s if len(s) > 1 else s + s for s in data.train]
    queries = data.test[:NQ]
    qn = jaccard_topk(queries, data.train, K)
    cached = sorted({s for nb in qn for s in nb})
    # neighbors WITHIN the cached set (training queries are cached sessions)
    csess = [sessions[c] for c in cached]
    nbr_c = jaccard_topk(csess, csess, K)
    nbr_c = np.asarray([nb if len(nb) == K else nb + [nb[-1]]*(K-len(nb)) for nb in nbr_c])
    print(f'[{ds}] cached={len(cached)} neighbor-matrix {nbr_c.shape}', flush=True)
    for cname, ckpt in cons:
        for R in (1, float('inf')):
            for seed in (0, 1, 2):
                torch.manual_seed(seed); rng = np.random.default_rng(seed)
                model = load_consumer(data, ckpt); model.train()
                opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
                emb0 = model.embedding.weight.detach().numpy()
                P_cache = pool(emb0, csess)
                for step in range(1, STEPS + 1):
                    if R != float('inf') and step % int(R) == 0:
                        ec = model.embedding.weight.detach().numpy()
                        P_cache = pool(ec, csess)
                    js = rng.integers(0, len(cached), BS)
                    prof = torch.from_numpy(P_cache[nbr_c[js]]).float()
                    batch = [csess[int(j)] for j in js]
                    L = max(len(s) for s in batch)
                    seq = torch.zeros(BS, L-1, dtype=torch.long); tgt = torch.zeros(BS, dtype=torch.long)
                    for b, s in enumerate(batch):
                        seq[b, :len(s)-1] = torch.tensor(s[:-1]); tgt[b] = s[-1]
                    out_f = run_forward(model, seq, prof)
                    lg = (out_f[0] if isinstance(out_f, tuple) else out_f).reshape(BS, -1).float()
                    loss = F.cross_entropy(lg[:, :data.n_items], tgt)
                    opt.zero_grad(); loss.backward(); opt.step()
                ec = model.embedding.weight.detach().numpy()
                # endpoint eval: retrieval with STALE cache vs current-table GT
                Qp = norm(pool(ec, queries)); Pc = pool(ec, sessions)
                cmask = np.zeros(len(sessions), dtype=bool); cmask[cached] = True
                Pst = Pc.copy(); Pst[cmask] = P_cache
                gt = top20(Qp, norm(Pc)); pred = top20(Qp, norm(Pst))
                rec = float(np.mean([len(set(gt[i]) & set(pred[i]))/20 for i in range(NQ)]))
                key = f'{ds}|{cname}|R={"inf" if R == float("inf") else int(R)}'
                results.setdefault(key, []).append(round(rec, 4))
                print(f'[{int(time.perf_counter()-t00)}s] {key} s{seed}: R@20 {rec:.4f}', flush=True)
summary = {k: {'mean': round(float(np.mean(v)), 4), 'std': round(float(np.std(v)), 4), 'seeds': v}
           for k, v in results.items()}
out = {'protocol': f'{EPOCHS}ep x {STEPS} steps x bs{BS}, k={K}, {NQ} queries, 3 seeds; '
                   'consumer trained with neighbor profiles from a Tier-1 cache, refresh cadence R; '
                   'endpoint eval = retrieval Recall@20 with the stale cache vs current-table GT',
       'results': summary,
       'scope': 'second-CONSUMER-CONFIGURATION replication (cross-attention ablation variants)'}
(RESULTS/'b7').mkdir(parents=True, exist_ok=True)
(RESULTS/'b7'/'consumer_replication.json').write_text(json.dumps(out, indent=1))
print(json.dumps(summary, indent=1))
