"""PIHPS — memeriksa temuan telaah panel 30 Okt 2026: (1) kalender hari libur, (2) skala rupiah/persen per komoditas,
(3) kepekaan CI h = 28 terhadap panjang blok, (4) sel sasaran hasil isi maju. Hanya baca.
Pakai: python 09_periksa_telaah.py > periksa_telaah.log
"""
import os

import numpy as np
import pandas as pd

T = os.path.join(os.environ.get('PIHPS_DATA', 'C:/Data/PIHPS'), 'turunan')
KONTEKS, BOOT = 512, 1000
Y = np.load(os.path.join(T, 'Y.npy'))
asal = pd.read_csv(os.path.join(T, 'asal.csv')).indeks.to_numpy()
tgl = pd.to_datetime(pd.read_csv(os.path.join(T, 'tanggal.csv')).tanggal)
der = pd.read_csv(os.path.join(T, 'deret.csv'))
kom = der.komoditas.to_numpy()
hari = pd.bdate_range(tgl.min(), tgl.max())
print(f'(1) tanggal {len(tgl)}, Senin-Jumat dalam rentang {len(hari)}, hilang {len(hari) - len(tgl)}')
libur = pd.to_datetime(['2024-04-08', '2024-04-09', '2024-04-10', '2024-04-11', '2024-04-12', '2024-04-15',
                        '2025-03-31', '2025-04-01', '2025-04-02', '2025-04-03', '2025-04-04', '2025-04-07',
                        '2026-03-19', '2026-03-20', '2026-03-23', '2026-03-24'])
ada = [str(d.date()) for d in libur if d in set(tgl)]
print(f'    tanggal cuti Lebaran yang ada di kalender: {len(ada)} dari {len(libur)}')
s = pd.Series(np.diff(Y, axis=1).ravel())
frac = []
for d in sorted(set(libur) & set(tgl)):
    i = tgl[tgl == d].index[0]
    x = Y[:, i] - Y[:, i - 1]
    frac.append(np.nanmean(x[np.isfinite(x)] == 0))
z = np.diff(Y[:, asal[0]:asal[-1] + 29], axis=1)
print(f'    bagian perubahan nol pada tanggal itu {np.mean(frac):.3f}; seluruh evaluasi {np.mean(z[np.isfinite(z)] == 0):.3f}')
print(f'    kelipatan Rp 50 di Y: {np.mean(np.mod(Y[np.isfinite(Y)], 50) == 0):.5f}')

P = {m: np.load(os.path.join(T, f'pred_{m}.npy')) for m in ['chronos2', 'timesfm', 'moirai', 'naif', 'ets']}
skala = np.array([[np.nanmean(np.abs(np.diff(Y[i, o - KONTEKS + 1:o + 1]))) for o in asal] for i in range(Y.shape[0])])
for h in [7, 28]:
    y = Y[:, asal + h]
    ok = np.isfinite(y) & (skala > 0)
    for m in P:
        ok &= np.isfinite(P[m][:, :, h - 1, 1])
    e = {m: np.where(ok, np.abs(y - P[m][:, :, h - 1, 1]), np.nan) for m in P}
    pct = {m: e[m] / y * 100 for m in P}
    print(f'\n(2) h = {h}: rerata |galat| % harga per komoditas; selisih rupiah terhadap naif; skala % harga')
    print(f'{"komoditas":22s} {"naif%":>7s} {"tfm%":>7s} {"moi%":>7s} {"ets%":>7s} {"d_tfm_Rp":>9s} {"skala%":>7s}')
    for k in list(np.unique(kom)) + ['SEMUA']:
        b = kom == k if k != 'SEMUA' else np.ones(len(kom), bool)
        print(f'{k:22s} {np.nanmean(pct["naif"][b]):7.3f} {np.nanmean(pct["timesfm"][b]):7.3f} {np.nanmean(pct["moirai"][b]):7.3f} '
              f'{np.nanmean(pct["ets"][b]):7.3f} {np.nanmean((e["timesfm"] - e["naif"])[b]):9.1f} '
              f'{np.nanmean((skala / Y[:, asal] * 100)[b]):7.3f}')
    d = {m: np.nanmean(np.where(ok, pct[m] - pct['naif'], np.nan), axis=0) for m in ['timesfm', 'moirai', 'chronos2']}
    rng = np.random.default_rng(20261030 + h)
    for bl in [4, 6, 8, 12]:
        nb = int(np.ceil(len(asal) / bl))
        idx = [np.concatenate([np.arange(b, b + bl) for b in rng.integers(0, len(asal) - bl + 1, nb)])[:len(asal)]
               for _ in range(BOOT)]
        t = []
        for m in d:
            bs = [np.nanmean(d[m][i]) for i in idx]
            t.append(f'{m} {np.nanmean(d[m]):+.3f} [{np.percentile(bs, 2.5):+.3f}; {np.percentile(bs, 97.5):+.3f}]')
        print(f'    selisih APE % vs naif, blok {bl}: ' + ' | '.join(t))
    me = {m: np.where(ok, e[m] / skala, np.nan) for m in P}
    for bl in [4, 6, 8, 12]:
        nb = int(np.ceil(len(asal) / bl))
        idx = [np.concatenate([np.arange(b, b + bl) for b in rng.integers(0, len(asal) - bl + 1, nb)])[:len(asal)]
               for _ in range(BOOT)]
        t = []
        for m in ['timesfm', 'moirai']:
            dl = np.nanmean(me[m], axis=0) - np.nanmean(me['naif'], axis=0)
            bs = [np.nanmean(dl[i]) for i in idx]
            t.append(f'{m} {np.nanmean(dl):+.3f} [{np.percentile(bs, 2.5):+.3f}; {np.percentile(bs, 97.5):+.3f}]')
        print(f'    selisih MASE vs naif, blok {bl}: ' + ' | '.join(t))
