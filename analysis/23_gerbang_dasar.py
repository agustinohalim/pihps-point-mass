"""PIHPS — peluang dasar komoditas (tanpa pengklasifikasi) pada 2022-2023, dan gerbang tanpa deret nasional.

Pelengkap 22_gerbang_telaah.py sesudah hasil 2024-2026 menunjukkan peluang dasar menyamai pengklasifikasi.
Pakai: venv-gpu python 23_gerbang_dasar.py > gerbang_dasar.log
"""
import importlib.util
import os

import numpy as np
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
s_ = importlib.util.spec_from_file_location('lintas', os.path.join(D, '19_gerbang_lintas_model.py'))
L = importlib.util.module_from_spec(s_)
s_.loader.exec_module(L)
P, G, T, tgl, HORIZON, KOLOM, kom = L.P, L.G, L.T, L.tgl, L.HORIZON, L.KOLOM, L.G.kom
wil = pd.read_csv(os.path.join(T, 'deret.csv')).wilayah.astype(str)
NAS = wil.str.lower().str.contains('semua|nasional|indonesia').to_numpy()


def dasar(a, n):
    d = np.array([a['lab'][kom == k][a['ok'][kom == k]].mean() for k in kom])
    return np.repeat(d[:, None], n, 1)


def main():
    pra = pd.read_csv(os.path.join(T, 'asal_pra.csv')).indeks.to_numpy()
    uji = pd.read_csv(os.path.join(T, 'asal.csv')).indeks.to_numpy()
    print('deret nasional', int(NAS.sum()), sorted(set(wil[NAS])))
    Qp, Qu = np.load(os.path.join(T, 'pred_timesfm_pra.npy')), np.load(os.path.join(T, 'pred_timesfm.npy'))
    for h in HORIZON:
        a, b = G.siapkan(pra, Qp, h), G.siapkan(uji, Qu, h)
        c, q = b['c'], b['q']
        gm = G.median_campuran(P.peluang(P.latih(a, KOLOM), b, KOLOM), c, q)
        bd = G.median_campuran(dasar(a, len(uji)), c, q)
        bn = dict(b)
        bn['ok'] = b['ok'] & ~NAS[:, None]
        idx = G.boot(len(uji), 20262500 + h)
        print(f'h={h} tanpa nasional: gerbang lawan naif {P.bedakan(gm, bn, c, idx)}; peluang dasar lawan naif {P.bedakan(bd, bn, c, idx)}')
    print('\n===== 97 asal 2022-2023')
    batas = np.datetime64('2022-01-01')
    for nama, s in (('TimesFM', 'timesfm'), ('Moirai-2', 'moirai')):
        Q = np.load(os.path.join(T, f'pred_{s}_pra.npy'))
        for h in HORIZON:
            d = G.siapkan(pra, Q, h)
            ta, ts = tgl.to_numpy()[pra], tgl.to_numpy()[pra + h]
            kl, ku = np.flatnonzero(ts < batas), np.flatnonzero(ta >= batas)
            a, b = P.potong(d, kl), P.potong(d, ku)
            b['ok'] = b['ok'] & np.isfinite(b['q']).all(2)
            c, q = b['c'], b['q']
            gm = G.median_campuran(P.peluang(P.latih(a, KOLOM), b, KOLOM), c, q)
            bd = G.median_campuran(dasar(a, len(ku)), c, q)
            idx = G.boot(len(ku), 20262600 + h)
            print(f'{nama} h={h}: peluang dasar lawan naif {P.bedakan(bd, b, c, idx)}')
            print(f'   gerbang lawan peluang dasar {P.bedakan(gm, b, bd, idx)}; peluang dasar lawan model {P.bedakan(bd, b, q[:, :, 1], idx)}')


if __name__ == '__main__':
    main()
