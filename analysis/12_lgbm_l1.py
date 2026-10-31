"""PIHPS — LightGBM dengan objektif L1 (ditambahkan sesudah telaah) lawan naif dan model fondasi, sasaran bersih, bootstrap
blok 8 asal, 10.000 ulangan; per kelompok komoditas. Pakai: python 12_lgbm_l1.py > lgbm_l1.log
"""
import importlib.util
import os

import numpy as np

s = importlib.util.spec_from_file_location('revisi', os.path.join(os.path.dirname(os.path.abspath(__file__)), '11_revisi.py'))
r = importlib.util.module_from_spec(s)
s.loader.exec_module(r)
P = {m: np.load(os.path.join(r.T, f'pred_{m}.npy'))[:, :, :, 1] for m in ['naif', 'lgbml1', 'timesfm', 'moirai', 'chronos2']}
skala = np.array([[np.nanmean(np.abs(np.diff(r.Y[i, o - r.KONTEKS + 1:o + 1]))) for o in r.asal] for i in range(r.n)])
for h in r.HORIZON:
    y = r.Y[:, r.asal + h]
    ok = np.isfinite(y) & (skala > 0) & r.survei[r.asal + h][None, :] & r.A[:, r.asal + h]
    ae = {m: np.where(ok, np.abs(y - P[m][:, :, h - 1]), np.nan) for m in P}
    me = {m: ae[m] / skala for m in P}
    ap = {m: ae[m] / y * 100 for m in P}
    idx = r.indeks_boot(20261030 + h, r.no)
    for nama, ks in (('semua', list(np.unique(r.kom))), ('bergerak', r.BERGERAK), ('lengket', r.LENGKET)):
        b = np.isin(r.kom, ks)
        print(f'h = {h} {nama}: ' + ' | '.join(
            f'{m} - {lawan} MASE {r.teks(r.ci(np.nanmean((me[m] - me[lawan])[b], axis=0), idx))} MAPE {r.teks(r.ci(np.nanmean((ap[m] - ap[lawan])[b], axis=0), idx))}'
            for m, lawan in (('lgbml1', 'naif'), ('timesfm', 'lgbml1'), ('moirai', 'lgbml1'))))
