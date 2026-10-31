"""PIHPS — apakah rugi model fondasi pada deret lengket hanya akibat harga PIHPS yang selalu kelipatan Rp 50?

Permintaan telaah 31 Okt 2026 (ketiga). Prakiraan median TimesFM dan Moirai-2 diubah dua cara: (a) dibulatkan ke
kelipatan Rp 50 terdekat; (b) ditahan di harga asal bila jaraknya kurang dari satu langkah Rp 50, selain itu (a).
Dinilai pada sasaran bersih 2024–2026 seperti 11_revisi.py, selisih terhadap naif, bootstrap blok 8.

Pakai: python 15_bulat50.py > bulat50.log
"""
import os

import numpy as np
import pandas as pd

T = os.path.join(os.environ.get('PIHPS_DATA', 'C:/Data/PIHPS'), 'turunan')
KONTEKS, BOOT, BLOK = 512, 10000, 8
BERGERAK = ['bawang_merah', 'cabai_merah_keriting', 'cabai_rawit_merah']
LENGKET = ['beras_medium_1', 'gula_pasir_lokal', 'minyak_goreng_curah']

Y = np.load(os.path.join(T, 'Y.npy'))
A = np.load(os.path.join(T, 'asli.npy'))
asal = pd.read_csv(os.path.join(T, 'asal.csv')).indeks.to_numpy()
tgl = pd.to_datetime(pd.read_csv(os.path.join(T, 'tanggal.csv')).tanggal)
kom = pd.read_csv(os.path.join(T, 'deret.csv')).komoditas.to_numpy()
survei = ~tgl.isin(pd.to_datetime(pd.read_csv(os.path.join(T, 'tanpa_survei.csv')).tanggal)).to_numpy()
n, no = Y.shape[0], len(asal)
skala = np.array([[np.nanmean(np.abs(np.diff(Y[i, o - KONTEKS + 1:o + 1]))) for o in asal] for i in range(n)])
rng = np.random.default_rng(20261031)
aw = rng.integers(0, no - BLOK + 1, (BOOT, int(np.ceil(no / BLOK))))
idx = (aw[:, :, None] + np.arange(BLOK)).reshape(BOOT, -1)[:, :no]


def ci(x):
    b = np.nanmean(x[idx], axis=1)
    return f'{np.nanmean(x):+.3f} [{np.percentile(b, 2.5):+.3f}; {np.percentile(b, 97.5):+.3f}]'


def main():
    print('kelipatan Rp 50 pada data:', bool(np.all(np.nan_to_num(Y) % 50 == 0)))
    y0 = Y[:, asal]
    for m in ('timesfm', 'moirai'):
        P = np.load(os.path.join(T, f'pred_{m}.npy'))[:, :, :, 1]
        for h in (7, 14, 28):
            y = Y[:, asal + h]
            ok = np.isfinite(y) & (skala > 0) & survei[asal + h][None, :] & A[:, asal + h]
            p = P[:, :, h - 1].astype(float)
            bulat = np.round(p / 50) * 50
            tahan = np.where(np.abs(p - y0) < 50, y0, bulat)
            e = lambda f: np.where(ok, np.abs(y - f), np.nan)
            for nama, f in (('asli', p), ('bulat', bulat), ('tahan', tahan)):
                dm = (e(f) - e(y0)) / skala
                da = (e(f) - e(y0)) / y * 100
                g = lambda x, s: np.nanmean(x[np.isin(kom, s)], axis=0)
                print(f'{m} h={h} {nama:5s}: semua MASE {ci(np.nanmean(dm, axis=0))} MAPE {ci(np.nanmean(da, axis=0))}'
                      f' | lengket MASE {ci(g(dm, LENGKET))} MAPE {ci(g(da, LENGKET))} | bergerak MAPE {ci(g(da, BERGERAK))}')


if __name__ == '__main__':
    main()
