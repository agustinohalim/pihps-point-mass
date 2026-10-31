"""PIHPS — kelengketan harga per komoditas: bagian perubahan harian yang tepat nol dan rerata |perubahan| %,
pada tanggal yang dicakup evaluasi (asal pertama s.d. sasaran terjauh). Termasuk sel hasil isi maju <= 3 hari,
yang sedikit menaikkan bagian nol. Pakai: python 07_kelengketan.py > kelengketan.log
"""
import os

import numpy as np
import pandas as pd

T = os.path.join(os.environ.get('PIHPS_DATA', 'C:/Data/PIHPS'), 'turunan')
Y = np.load(os.path.join(T, 'Y.npy'))
asal = pd.read_csv(os.path.join(T, 'asal.csv')).indeks.to_numpy()
kom = pd.read_csv(os.path.join(T, 'deret.csv')).komoditas.to_numpy()
Z = Y[:, asal[0]:asal[-1] + 29]
d = np.diff(Z, axis=1)
r = np.abs(d) / Z[:, :-1] * 100
print(f'semua: nol {np.mean(d[np.isfinite(d)] == 0) * 100:.1f} %, |perubahan| {np.nanmean(r):.3f} %')
for k in np.unique(kom):
    m = kom == k
    f = np.isfinite(d[m])
    print(f'{k}: nol {np.mean(d[m][f] == 0) * 100:.1f} %, |perubahan| {np.nanmean(r[m]):.3f} %')
