"""Uji komoditas tak tersentuh (rancangan di Hasil_Uji_PIHPS.md, "Rancangan skrip 29"): persiapan sesudah 02_siapkan.

Dijalankan dengan PIHPS_DATA=C:/Data/PIHPS_lain sesudah `01_unduh.py` (PIHPS_KOMODITAS=lain) dan `02_siapkan.py`.
Memeriksa bahwa kalender sama dengan data naskah, lalu menyalin asal.csv, asal_pra.csv, dan tanpa_survei.csv dari
data naskah apa adanya (beku), dan menulis asli.npy dengan aturan yang sama seperti 10_kalender.py.

Pakai: PIHPS_DATA=C:/Data/PIHPS_lain python 29a_siapkan_lain.py > siapkan_lain.log
"""
import os
import shutil

import numpy as np
import pandas as pd

ASLI = os.path.join('C:/Data/PIHPS', 'turunan')
D = os.environ['PIHPS_DATA']
T = os.path.join(D, 'turunan')
assert os.path.normcase(os.path.abspath(T)) != os.path.normcase(os.path.abspath(ASLI))


def main():
    t0 = pd.read_csv(os.path.join(ASLI, 'tanggal.csv')).tanggal
    t1 = pd.read_csv(os.path.join(T, 'tanggal.csv')).tanggal
    print(f'kalender naskah {len(t0)}, komoditas lain {len(t1)}, sama: {t0.equals(t1)}')
    if not t0.equals(t1):
        print('  hanya di naskah:', sorted(set(t0) - set(t1))[:20])
        print('  hanya di lain:', sorted(set(t1) - set(t0))[:20])
        raise SystemExit('kalender berbeda: berhenti, rancangan menuntut asal yang sama')
    for f in ('asal.csv', 'asal_pra.csv', 'tanpa_survei.csv'):
        shutil.copyfile(os.path.join(ASLI, f), os.path.join(T, f))
    Y = np.load(os.path.join(T, 'Y.npy'))
    der = pd.read_csv(os.path.join(T, 'deret.csv'))
    tgl = pd.to_datetime(t1)
    d = pd.read_csv(os.path.join(D, 'harga.csv'), parse_dates=['tanggal'])
    d = d[d.wilayah != 'Dum']
    kunci = {(k, w): i for i, (k, w) in enumerate(zip(der.komoditas, der.wilayah))}
    ti = {t: j for j, t in enumerate(tgl)}
    A = np.zeros(Y.shape, bool)
    for k, w, t, h in zip(d.komoditas, d.wilayah, d.tanggal, d.harga):
        i = kunci.get((k, w))
        if i is not None and np.isfinite(h):
            A[i, ti[t]] = True
    np.save(os.path.join(T, 'asli.npy'), A)
    asal = pd.read_csv(os.path.join(T, 'asal.csv')).indeks.to_numpy()
    print(f'deret {Y.shape[0]}; sel asli rentang evaluasi {A[:, asal[0]:asal[-1] + 29].mean():.4f}')
    print(der.komoditas.value_counts().sort_index().to_string())


if __name__ == '__main__':
    main()
