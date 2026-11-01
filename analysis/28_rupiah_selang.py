"""PIHPS — selang bootstrap untuk selisih MAE rupiah per kelompok. Rancangan di Hasil_Uji_PIHPS.md ("Rancangan skrip 28").

Pakai: venv-gpu python 28_rupiah_selang.py > rupiah_selang.log
"""
import importlib.util
import os

import numpy as np
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
s_ = importlib.util.spec_from_file_location('cd', os.path.join(D, '24_campuran_dasar.py'))
C = importlib.util.module_from_spec(s_)
s_.loader.exec_module(C)
G, T, kom, dasar, HORIZON = C.G, C.T, C.kom, C.dasar, C.HORIZON


def main():
    pra = pd.read_csv(os.path.join(T, 'asal_pra.csv')).indeks.to_numpy()
    uji = pd.read_csv(os.path.join(T, 'asal.csv')).indeks.to_numpy()
    Qp, Qu = np.load(os.path.join(T, 'pred_timesfm_pra.npy')), np.load(os.path.join(T, 'pred_timesfm.npy'))
    for h in HORIZON:
        a, b = G.siapkan(pra, Qp, h), G.siapkan(uji, Qu, h)
        ok, c, q, y = b['ok'], b['c'], b['q'], b['y']
        pm, _ = dasar(a, len(uji))
        x, m = G.median_campuran(pm, c, q), q[:, :, 1]
        idx = G.boot(len(uji), 20262700 + h)
        e = lambda f: np.where(ok, np.abs(y - f), np.nan)
        s = lambda f: np.where(ok, 200 * np.abs(y - f) / (np.abs(y) + np.abs(f)), np.nan)
        print(f'===== h={h}; sMAPE naif {np.nanmean(s(c)):.3f}, TimesFM {np.nanmean(s(m)):.3f}, campuran {np.nanmean(s(x)):.3f}')
        for nama, kel in (('bergerak', C.BERGERAK), ('lengket', C.LENGKET)):
            r = np.isin(kom, kel)
            for lbl, p, w in (('campuran-TimesFM', x, m), ('campuran-naif', x, c), ('TimesFM-naif', m, c)):
                d = (e(p) - e(w))[r]
                print(f'  {nama} {lbl}: gabungan {np.nanmean(d):+.2f}; rerata asal {G.ci(np.nanmean(d, 0), idx)}', flush=True)


if __name__ == '__main__':
    main()
