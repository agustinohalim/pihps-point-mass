"""PIHPS — gerbang empat ciri (tanpa masukan model fondasi) dipasang pada kuantil tiap model.

Post hoc: rancangan ditulis di Hasil_Uji_PIHPS.md ("Rancangan gerbang lintas model") sesudah hasil ablasi, sebelum skrip
ini dijalankan. Pengklasifikasi sama dengan 16_gerbang.py, hanya kolom masukannya kelengketan, volatilitas, komoditas,
dan jarak ke Idulfitri; dilatih satu kali, dibekukan, peluangnya dipakai untuk semua model.

Pakai: venv-gpu python 19_gerbang_lintas_model.py > gerbang_lintas.log
"""
import importlib.util
import os

import numpy as np
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))


def muat(nama, berkas):
    s = importlib.util.spec_from_file_location(nama, os.path.join(D, berkas))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


P = muat('periksa', '18_periksa_gerbang.py')
G, T, tgl, HORIZON, KOLOM = P.G, P.T, P.tgl, P.HORIZON, P.TANPA_MODEL
MODEL = [('Chronos-2', 'chronos2'), ('Chronos-Bolt', 'chronosbolt'), ('TimesFM 2.5', 'timesfm'), ('Moirai-2', 'moirai'),
         ('Chronos-2 dilatih', 'chronos2_ft'), ('Chronos-2 dilatih+kov', 'chronos2_ftkov')]


def baris(nama, Q, b, p, idx):
    q = np.sort(Q.astype(float), axis=2)
    gm = G.median_campuran(p, b['c'], q)
    print(f'  {nama}')
    print('    saja lawan naif:     ' + P.bedakan(q[:, :, 1], b, b['c'], idx))
    print('    bergerbang lawan naif:' + P.bedakan(gm, b, b['c'], idx))
    print('    bergerbang lawan saja:' + P.bedakan(gm, b, q[:, :, 1], idx), flush=True)


def main():
    pra = pd.read_csv(os.path.join(T, 'asal_pra.csv')).indeks.to_numpy()
    uji = pd.read_csv(os.path.join(T, 'asal.csv')).indeks.to_numpy()
    Qp, Qu = np.load(os.path.join(T, 'pred_timesfm_pra.npy')), np.load(os.path.join(T, 'pred_timesfm.npy'))
    print('===== 142 asal 2024-2026, gerbang empat ciri dilatih pada 202 asal pra-2024')
    for h in HORIZON:
        a, b = G.siapkan(pra, Qp, h), G.siapkan(uji, Qu, h)
        p = P.peluang(P.latih(a, KOLOM), b, KOLOM)
        idx = G.boot(len(uji), 20261700 + h)
        print(f'h={h}')
        for nama, s in MODEL:
            baris(nama, np.load(os.path.join(T, f'pred_{s}.npy'))[:, :, h - 1, :], b, p, idx)
    print('\n===== 97 asal 2022-2023, gerbang empat ciri dilatih pada sasaran < 2022')
    batas = np.datetime64('2022-01-01')
    for nama, s in (('TimesFM 2.5', 'timesfm'), ('Moirai-2', 'moirai')):
        Q = np.load(os.path.join(T, f'pred_{s}_pra.npy'))
        for h in HORIZON:
            d = G.siapkan(pra, Q, h)
            ta, ts = tgl.to_numpy()[pra], tgl.to_numpy()[pra + h]
            kl, ku = np.flatnonzero(ts < batas), np.flatnonzero(ta >= batas)
            a, b = P.potong(d, kl), P.potong(d, ku)
            b['ok'] = b['ok'] & np.isfinite(b['q']).all(2)
            p = P.peluang(P.latih(a, KOLOM), b, KOLOM)
            print(f'h={h}')
            baris(nama, b['q'], b, p, G.boot(len(ku), 20261800 + h))


if __name__ == '__main__':
    main()
