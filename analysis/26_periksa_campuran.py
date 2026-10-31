"""PIHPS — dua pemeriksaan atas campuran sesudah telaah 31d: horizon 1 hari, dan kepekaan terhadap massa p.

Massa: p latih dikali 0,5 dan 1,5, serta p amatan 2024-2026 (peramal tidak mungkin tahu; batas atas yang layak).
Pakai: venv-gpu python 26_periksa_campuran.py > periksa_campuran.log
"""
import importlib.util
import os

import numpy as np
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
s_ = importlib.util.spec_from_file_location('cd', os.path.join(D, '24_campuran_dasar.py'))
C = importlib.util.module_from_spec(s_)
s_.loader.exec_module(C)
P, G, L, T, kom, dasar, rmae = C.P, C.G, C.L, C.T, C.kom, C.dasar, C.rmae


def main():
    pra = pd.read_csv(os.path.join(T, 'asal_pra.csv')).indeks.to_numpy()
    uji = pd.read_csv(os.path.join(T, 'asal.csv')).indeks.to_numpy()
    Qp, Qu = np.load(os.path.join(T, 'pred_timesfm_pra.npy')), np.load(os.path.join(T, 'pred_timesfm.npy'))

    a, b = G.siapkan(pra, Qp, 1), G.siapkan(uji, Qu, 1)
    pm, _ = dasar(a, len(uji))
    idx = G.boot(len(uji), 20263100)
    print('===== h=1, 142 asal')
    for nama, s in L.SEMUA:
        Q = np.sort(np.load(os.path.join(T, f'pred_{s}.npy'))[:, :, 0, :].astype(float), axis=2)
        m, c = Q[:, :, 1], b['c']
        x = G.median_campuran(pm, c, Q)
        print(f'  {nama}: saja lawan naif {P.bedakan(m, b, c, idx)}')
        print(f'    campuran lawan naif {P.bedakan(x, b, c, idx)}; tepat {np.mean((x == c)[b["ok"]]) * 100:.1f} %')

    for h in (7, 14, 28):
        a, b = G.siapkan(pra, Qp, h), G.siapkan(uji, Qu, h)
        pm, _ = dasar(a, len(uji))
        c, q, ok = b['c'], b['q'], b['ok']
        amat = {k: float(b['lab'][kom == k][ok[kom == k]].mean()) for k in np.unique(kom)}
        pa = np.repeat(np.array([amat[k] for k in kom])[:, None], len(uji), 1)
        idx = G.boot(len(uji), 20263200 + h)
        print(f'===== TimesFM h={h}, kepekaan p')
        dasar_x = G.median_campuran(pm, c, q)
        print(f'  p latih lawan naif: {P.bedakan(dasar_x, b, c, idx)}')
        for nama, r in (('p x 0,5', pm * .5), ('p x 1,5', np.minimum(pm * 1.5, .95)), ('p amatan', pa)):
            x = G.median_campuran(r, c, q)
            print(f'  {nama}: lawan naif {P.bedakan(x, b, c, idx)}; lawan p latih {P.bedakan(x, b, dasar_x, idx)}', flush=True)


if __name__ == '__main__':
    main()
