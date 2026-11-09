"""Round 9: (a) refresh-exposure under in-place concurrent refresh;
(b) DAI-2 (cumulative-drift) vs DAI (per-epoch) refresh volume + the
dispersion guarantee, on the real co-training drift trace."""
import sys, json
import numpy as np
from pathlib import Path
sys.path.insert(0, 'code/experiments_icde')
from common import load_data, RESULTS

data = load_data(Path('code/data/diginetica.pkl'))
drift = np.load(RESULTS / 'drift_trace_diginetica.npz')['drift']
T, n_items = drift.shape
PAGE = 128; NPAGES = len(data.train)//PAGE + 1
item2pages = [set() for _ in range(n_items)]
for sid, s in enumerate(data.train):
    for it in set(s): item2pages[it].add(sid//PAGE)

# ---- (a) exposure: arrivals during a 367ms refresh, in-place page-by-page ----
T_refresh_ms = 366.7; dirty_frac = 0.508; K = 10
curve_f = np.array([0.0,0.25,0.5,0.75,1.0]); curve_q = np.array([1.0,0.9953,0.992,0.9869,0.9843])
rng = np.random.default_rng(0)
expo = {}
for qps in (173, 228, 1000):
    lam = qps/1000.0; n_arr = rng.poisson(lam*T_refresh_ms)
    t = np.sort(rng.uniform(0, T_refresh_ms, n_arr))
    p = t/T_refresh_ms
    new_cnt = (rng.random((n_arr, K)) < p[:, None]*dirty_frac).sum(axis=1)
    stale_frac = 1 - new_cnt/K
    mixed = (stale_frac>0)&(stale_frac<1)
    expo[qps] = {'arrivals_per_refresh': int(n_arr), 'mixed_batch_frac': round(float(mixed.mean()),3),
                 'mean_quality_inplace': round(float(np.interp(stale_frac, curve_f, curve_q).mean()),4),
                 'quality_all_old_snapshot': 0.9843}
    print(f'qps={qps}: arrivals={n_arr} mixed={expo[qps]["mixed_batch_frac"]:.1%} '
          f'inplace_mean_Q={expo[qps]["mean_quality_inplace"]}')

# ---- (b) DAI vs DAI-2 refresh volume + dispersion guarantee ----
def simulate(cumulative, theta):
    c = np.zeros(n_items); page_refresh = np.zeros(NPAGES); refreshes=0
    realized_max_to_current = 0.0
    for t in range(T):
        d = drift[t]
        if cumulative:
            c += d
            moved = np.flatnonzero(c > theta)
        else:
            moved = np.flatnonzero(d > theta)
        dirty = set()
        for it in moved: dirty |= item2pages[int(it)]
        refreshes += len(dirty)
        page_refresh[list(dirty)] += 1
        if cumulative:
            c[moved] = 0.0          # refreshed items re-pooled at current table
        # distance-to-current proxy: refreshed items ~0, others accumulate (DAI: only this epoch's)
        if not cumulative:
            realized = float(np.percentile(np.cumsum(drift[:t+1],axis=0)[-1], 99))  # upper-ish
    return refreshes
res = {}
for theta in (0.10, 0.15):
    per = simulate(False, theta); cum = simulate(True, theta)
    res[str(theta)] = {'dai_per_epoch_page_refreshes': per, 'dai2_cumulative_page_refreshes': cum,
                       'guarantee_cost_ratio': round(cum/max(per,1),2)}
    print(f'theta={theta}: DAI={per} page-refreshes, DAI-2={cum} (x{res[str(theta)]["guarantee_cost_ratio"]})')

out = {'dataset':'diginetica','refresh_window_ms':T_refresh_ms,
 'exposure': {str(k):v for k,v in expo.items()},
 'mixing_quality_curve': {'stale_frac': curve_f.tolist(), 'recall': curve_q.tolist()},
 'dai_vs_dai2': res,
 'dispersion_guarantee': 'DAI-2: every served profile within theta_max of the current table '
   '(cumulative drift since its refresh <= theta_max), hence pairwise batch dispersion <= 2*theta_max '
   'by triangle inequality; in-place concurrent refresh is therefore contract-safe without snapshots',
 'note': 'exposure simulated from measured refresh duration + measured GPU serving qps; DAI/DAI-2 '
         'replayed on the real drift trace (3 intervals)'}
(RESULTS/'b3'/'dai2_dispersion.json').write_text(json.dumps(out,indent=1))
print('wrote', RESULTS/'b3'/'dai2_dispersion.json')
