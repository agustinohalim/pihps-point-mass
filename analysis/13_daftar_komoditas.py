"""Audit: apakah aturan kelengketan per deret memberi lebih dari daftar komoditas tetap? Walk-forward, h 7/14/28."""
import os

import numpy as np, pandas as pd
T = os.path.join(os.environ.get('PIHPS_DATA', 'C:/Data/PIHPS'), 'turunan', '')
Y = np.load(T + 'Y.npy'); A = np.load(T + 'asli.npy')
asal = pd.read_csv(T + 'asal.csv').indeks.to_numpy()
tgl = pd.to_datetime(pd.read_csv(T + 'tanggal.csv').tanggal)
kom = pd.read_csv(T + 'deret.csv').komoditas.to_numpy()
ts = set(pd.to_datetime(pd.read_csv(T + 'tanpa_survei.csv').tanggal))
sv = np.array([t not in ts for t in tgl])
n, no = Y.shape[0], len(asal)
sk = np.array([[np.nanmean(np.abs(np.diff(Y[i, o - 511:o + 1]))) for o in asal] for i in range(n)])
tf = np.load(T + 'pred_timesfm.npy')[:, :, :, 1]; na = np.load(T + 'pred_naif.npy')[:, :, :, 1]
hs = np.flatnonzero(sv); z = np.full((n, no), np.nan)
for k, o in enumerate(asal):
    j = hs[hs <= o][-61:]; d = np.diff(Y[:, j], axis=1)
    z[:, k] = np.nansum(d == 0, axis=1) / np.isfinite(d).sum(axis=1)
BURN, BLOK, B = 52, 8, 10000
ev = np.arange(BURN, no)
rng = np.random.default_rng(20261031)
aw = rng.integers(0, len(ev) - BLOK + 1, (B, int(np.ceil(len(ev) / BLOK))))
idx = (aw[:, :, None] + np.arange(BLOK)).reshape(B, -1)[:, :len(ev)]
def ci(x):
    b = np.nanmean(x[idx], axis=1); return f'{np.nanmean(x):+.3f} [{np.percentile(b,2.5):+.3f}; {np.percentile(b,97.5):+.3f}]'
kk = np.unique(kom)
for h in (7, 14, 28):
    y = Y[:, asal + h]; ok = np.isfinite(y) & sv[asal + h][None] & A[:, asal + h]
    err = lambda p: np.where(ok, np.abs(y - p), np.nan)
    mt, mn = err(tf[:, :, h - 1]) / sk, err(na[:, :, h - 1]) / sk
    at, an = err(tf[:, :, h - 1]) / y * 100, err(na[:, :, h - 1]) / y * 100
    lag = int(np.ceil(h / 5))
    R = {x: np.full((n, no), np.nan) for x in ('mz', 'az', 'mk', 'ak')}
    pk = np.full((len(kk), no), np.nan)
    for k in range(BURN, no):
        l = np.arange(0, k - lag + 1)
        gr = np.arange(.2, 1.01, .1)
        t = gr[int(np.nanargmin([np.nanmean(np.where(z[:, l] < g, mt[:, l], mn[:, l])) for g in gr]))]
        cz = z[:, k] < t
        # daftar: seluruh komoditas memakai TimesFM bila MASE pooled-nya lebih rendah pada asal lampau
        ck = np.zeros(n, bool)
        for q, c in enumerate(kk):
            b = kom == c; u = np.nanmean(mt[b][:, l]) < np.nanmean(mn[b][:, l]); ck[b] = u; pk[q, k] = u
        R['mz'][:, k], R['az'][:, k] = np.where(cz, mt[:, k], mn[:, k]), np.where(cz, at[:, k], an[:, k])
        R['mk'][:, k], R['ak'][:, k] = np.where(ck, mt[:, k], mn[:, k]), np.where(ck, at[:, k], an[:, k])
    pm = lambda a, b: np.nanmean(a[:, ev] - b[:, ev], axis=0)
    print(f'h={h}: daftar vs naif MASE {ci(pm(R["mk"], mn))} MAPE {ci(pm(R["ak"], an))}')
    print(f'      kelengketan vs naif MASE {ci(pm(R["mz"], mn))} MAPE {ci(pm(R["az"], an))}')
    print(f'      kelengketan - daftar MASE {ci(pm(R["mz"], R["mk"]))} MAPE {ci(pm(R["az"], R["ak"]))}')
    print('      daftar pakai TimesFM: ' + ' '.join(f'{c[:8]}={np.nanmean(pk[q, ev]):.2f}' for q, c in enumerate(kk)))
