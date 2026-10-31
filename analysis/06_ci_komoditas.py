"""PIHPS — CI per komoditas untuk selisih MASE model fondasi terhadap naif (dan ETS).

Melengkapi 05_evaluasi.py, yang hanya memberi MASE per komoditas tanpa CI. Himpunan pasangan, skala
MASE, dan bootstrap blok sama persis dengan 05_evaluasi.py (blok 4 asal, 1.000 ulangan); bedanya rerata
per asal diambil hanya atas deret komoditas itu. Benih bootstrap sama per horizon, jadi indeks ulangan
sama untuk semua komoditas.

Banyak perbandingan (9 komoditas × 3 horizon × 4 model × 2 lawan): CI ini tanpa koreksi kelipatan,
dibaca sebagai pola, bukan uji satu per satu.

Pakai: python 06_ci_komoditas.py > ci_komoditas.log
"""
import os

import numpy as np
import pandas as pd

D = os.environ.get('PIHPS_DATA', 'C:/Data/PIHPS')
T = os.path.join(D, 'turunan')
KONTEKS, HORIZON, BOOT, BLOK = 512, [7, 14, 28], 1000, 4
FONDASI = ['chronos2', 'chronosbolt', 'timesfm', 'moirai']
SEMUA = FONDASI + ['naif', 'naifmusim', 'ets', 'arima', 'lgbm']


def main():
    Y = np.load(os.path.join(T, 'Y.npy'))
    asal = pd.read_csv(os.path.join(T, 'asal.csv')).indeks.to_numpy()
    kom = pd.read_csv(os.path.join(T, 'deret.csv')).komoditas.to_numpy()
    P = {m: np.load(os.path.join(T, f'pred_{m}.npy')) for m in SEMUA}
    n = Y.shape[0]
    skala = np.array([[np.nanmean(np.abs(np.diff(Y[i, o - KONTEKS + 1:o + 1]))) for o in asal] for i in range(n)])
    nol = skala == 0
    baris = []
    for h in HORIZON:
        y = Y[:, asal + h]
        ok = np.isfinite(y) & ~nol & np.isfinite(skala)
        for m in SEMUA:
            ok &= np.isfinite(P[m][:, :, h - 1, 1])
        mase = {m: np.where(ok, np.abs(y - P[m][:, :, h - 1, 1]) / skala, np.nan) for m in FONDASI + ['naif', 'ets']}
        rng = np.random.default_rng(20261028 + h)
        nb = int(np.ceil(len(asal) / BLOK))
        idx = [np.concatenate([np.arange(b, b + BLOK) for b in rng.integers(0, len(asal) - BLOK + 1, nb)])[:len(asal)]
               for _ in range(BOOT)]
        for k in np.unique(kom):
            pilih = kom == k
            pa = {m: np.nanmean(mase[m][pilih], axis=0) for m in mase}
            for m in FONDASI:
                for ref in ('naif', 'ets'):
                    dlt = pa[m] - pa[ref]
                    bs = [np.nanmean(dlt[i]) for i in idx]
                    lo, hi = np.percentile(bs, [2.5, 97.5])
                    baris.append({'h': h, 'komoditas': k, 'model': m, 'lawan': ref, 'd': np.nanmean(dlt),
                                  'lo': lo, 'hi': hi,
                                  'tanda': '<0' if hi < 0 else ('>0' if lo > 0 else '0')})
    t = pd.DataFrame(baris)
    t.to_csv(os.path.join(T, 'ci_komoditas.csv'), index=False)
    for h in HORIZON:
        for ref in ('naif', 'ets'):
            s = t[(t.h == h) & (t.lawan == ref)]
            print(f'\nh = {h}, lawan {ref}: selisih MASE [CI95] (negatif = model fondasi lebih baik)')
            tab = s.assign(sel=s.apply(lambda r: f'{r.d:+.3f} [{r.lo:+.3f}; {r.hi:+.3f}]', axis=1)) \
                   .pivot(index='komoditas', columns='model', values='sel')[FONDASI]
            print(tab.to_string())


if __name__ == '__main__':
    main()
