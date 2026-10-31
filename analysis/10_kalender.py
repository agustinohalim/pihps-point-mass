"""PIHPS — mengenali tanggal tanpa survei dari data sendiri, tanpa daftar libur yang ditulis tangan.

Temuan telaah 30 Okt 2026: kalender memuat hampir semua Senin–Jumat, termasuk hari libur, dan sel hari itu
tampak disalin dari hari sebelumnya. Di sini tanggal "tanpa survei" = tanggal yang bagian perubahan nol pada
kedua cabai (komoditas paling sering bergerak, normalnya sekitar 28 % nol) >= AMBANG. Ambang dipilih dari
sebaran bagian nol yang dicetak di bawah (tidak ada satu tanggal pun di 0,5–0,6: tanggal biasa dan tanggal
salinan terpisah bersih), bukan dari hasil model. Sel asli (sebelum isi maju <= 3 hari) dibaca dari harga.csv.

Keluaran di C:/Data/PIHPS/turunan: tanpa_survei.csv (tanggal, bagian nol cabai), asli.npy (deret × tanggal,
True = sel ada di unduhan mentah).

Pakai: python 10_kalender.py > kalender.log
"""
import os

import numpy as np
import pandas as pd

D = os.environ.get('PIHPS_DATA', 'C:/Data/PIHPS')
T = os.path.join(D, 'turunan')
AMBANG = 0.60


def main():
    Y = np.load(os.path.join(T, 'Y.npy'))
    der = pd.read_csv(os.path.join(T, 'deret.csv'))
    tgl = pd.to_datetime(pd.read_csv(os.path.join(T, 'tanggal.csv')).tanggal)
    asal = pd.read_csv(os.path.join(T, 'asal.csv')).indeks.to_numpy()
    cabai = der.komoditas.str.contains('Cabai', case=False).to_numpy()
    print(f'deret cabai {cabai.sum()}; komoditas: {sorted(der.komoditas.unique())}')
    dY = np.diff(Y[cabai], axis=1)
    nol = np.where(np.isfinite(dY), dY == 0, np.nan)
    frac = np.r_[np.nan, np.nanmean(nol, axis=0)]
    s = pd.Series(frac, index=tgl)
    print('sebaran bagian nol cabai per tanggal:')
    print(pd.cut(s.dropna(), [-.01, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99, 1]).value_counts().sort_index().to_string())
    print('sebaran pada rentang evaluasi:')
    ev = s.iloc[asal[0]:asal[-1] + 29]
    print(pd.cut(ev, [-.01, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99, 1]).value_counts().sort_index().to_string())
    tanpa = s[s >= AMBANG]
    print(f'\ntanggal tanpa survei (>= {AMBANG}): {len(tanpa)} seluruh data; {int((ev >= AMBANG).sum())} di rentang evaluasi')
    print(ev[ev >= AMBANG].round(3).to_string())
    print('\ntanggal rentang evaluasi dengan bagian nol 0,6–0,9 (zona abu-abu):')
    print(ev[(ev >= .6) & (ev < AMBANG)].round(3).to_string())
    tanpa.rename('nol_cabai').rename_axis('tanggal').to_csv(os.path.join(T, 'tanpa_survei.csv'))

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
    Z = A[:, asal[0]:asal[-1] + 29]
    print(f'\nsel asli rentang evaluasi {Z.mean():.4f}; sel Y terisi {np.isfinite(Y[:, asal[0]:asal[-1] + 29]).mean():.4f}')
    print('selisih harga pada tanggal tanpa survei (deret cabai, selain nol):',
          np.nanmean(np.abs(dY[:, np.isin(tgl.iloc[1:], tanpa.index)]) > 0).round(4))


if __name__ == '__main__':
    main()
