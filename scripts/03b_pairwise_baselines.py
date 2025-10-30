# python -m scripts.03b_pairwise_baselines
import sys, pickle, numpy as np, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from cmorph.distances import correlation_distance, ncc_max_distance, frechet_1d_distance

wins = pickle.load(open("artifacts/windows.pkl","rb"))
X = np.stack(wins)
N = X.shape[0]

def pairwise(metric_fn, name):
    D = np.zeros((N,N), float)
    t0 = time.time()
    for i in range(N):
        for j in range(i+1, N):
            D[i,j] = metric_fn(X[i], X[j])
    D = D + D.T
    np.fill_diagonal(D, 0.0)
    np.save(f"artifacts/D_{name}.npy", D)
    print(f"✅ artifacts/D_{name}.npy (t={time.time()-t0:.1f}s)")

pairwise(correlation_distance, "corr")
pairwise(lambda a,b: ncc_max_distance(a,b,max_lag=4), "ncc")
pairwise(frechet_1d_distance, "frechet")
