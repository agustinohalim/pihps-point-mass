"""PIHPS — dari mana kekalahan model lawan naif datang: jendela tak berubah neto (sasaran = harga asal) lawan jendela lain.

Rancangan ditulis sebelum dijalankan: Hasil_Uji_PIHPS.md, "Rancangan skrip 31". Deskriptif; metode tidak berubah.
Selisih galat model dikurangi naif dipecah menjadi dua sumbangan yang dijumlah tepat ke selisih rata-rata.

Pakai: venv-gpu python 31_mekanisme.py > mekanisme.log
       PIHPS_DATA=C:/Data/PIHPS_lain venv-gpu python 31_mekanisme.py > mekanisme_lain.log
"""
import importlib.util
import os

import numpy as np
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
s_ = importlib.util.spec_from_file_location('ta', os.path.join(D, '27_telaah_akhir.py'))
K = importlib.util.module_from_spec(s_)
s_.loader.exec_module(K)
P, G, T, kom, HORIZON = K.P, K.G, K.T, K.kom, K.HORIZON
LAIN_DATA = 'PIHPS_lain' in os.environ.get('PIHPS_DATA', '')
FONDASI = [('Chronos-2', 'chronos2'), ('Chronos-Bolt', 'chronosbolt'), ('TimesFM 2.5', 'timesfm'), ('Moirai-2', 'moirai')]
LATIH = [('LightGBM-L1', 'lgbml1'), ('ETS', 'ets'), ('ARIMA', 'arima'), ('Chronos-2 disetel', 'chronos2_ft')]
if LAIN_DATA:
    KEL = {'cabai': [k for k in np.unique(kom) if k.startswith('cabai')],
           'lain': [k for k in np.unique(kom) if not k.startswith('cabai')]}
else:
    KEL = {'lengket': K.LENGKET, 'bergerak': K.BERGERAK, 'lain': K.LAIN}


def pecah(x, b, idx, r=None):
    ok, y, c, sk, lab = b['ok'], b['y'], b['c'], b['skala'], b['lab'] == 1
    if r is not None:
        ok = ok & r
    d = np.abs(y - x) - np.abs(y - c)
    n = ok.sum(0).astype(float)
    n[n == 0] = np.nan
    keluar = []
    for nama, w in (('MASE', sk), ('MAPE', y / 100)):
        dd = np.where(ok, d / w, 0.0)
        tak = (dd * lab).sum(0) / n
        ber = (dd * ~lab).sum(0) / n
        keluar.append(f'{nama} total {G.ci(tak + ber, idx)} = tak berubah {G.ci(tak, idx)} + berubah {G.ci(ber, idx)}')
    return '; '.join(keluar) + f'; bagian tak berubah {np.nanmean(np.where(ok, lab, np.nan)) * 100:.1f} %'


def main():
    uji = pd.read_csv(os.path.join(T, 'asal.csv')).indeks.to_numpy()
    print(f'deret {len(kom)}, asal {len(uji)}, data {T}')
    model = FONDASI + ([] if LAIN_DATA else LATIH)
    for h in HORIZON:
        idx = G.boot(len(uji), 20263200 + h)
        print(f'\n===== h={h}')
        for nama, s in model:
            Q = np.load(os.path.join(T, f'pred_{s}.npy'))
            b = G.siapkan(uji, Q, h)
            # LightGBM-L1 hanya menyimpan median (kolom 1); np.sort di siapkan menggesernya, jadi median diambil mentah
            m = Q[:, :, h - 1, 1].astype(float)
            b['ok'] = b['ok'] & np.isfinite(m)
            print(f'  {nama:18s} semua: {pecah(m, b, idx)}')
            for nk, ks in KEL.items():
                r = np.isin(kom, ks)[:, None]
                print(f'  {"":18s} {nk:9s}: {pecah(m, b, idx, r)}', flush=True)


if __name__ == '__main__':
    main()
