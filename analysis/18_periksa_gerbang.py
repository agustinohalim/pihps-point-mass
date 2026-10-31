"""PIHPS — pemeriksaan gerbang diam-atau-gerak: periode lebih awal, ablasi, kalibrasi, pembanding Chronos-2 dilatih.

Rancangan ditulis di Hasil_Uji_PIHPS.md ("Rancangan pemeriksaan gerbang") sebelum skrip ini dijalankan. Bentuk dan
parameter gerbang sama persis dengan 16_gerbang.py.

Pakai: venv-gpu python 18_periksa_gerbang.py > periksa_gerbang.log
"""
import importlib.util
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

spek = importlib.util.spec_from_file_location('gerbang', os.path.join(os.path.dirname(os.path.abspath(__file__)), '16_gerbang.py'))
G = importlib.util.module_from_spec(spek)
spek.loader.exec_module(G)
T, kom, tgl, HORIZON = G.T, G.kom, G.tgl, G.HORIZON
BERGERAK = ['bawang_merah', 'cabai_merah_keriting', 'cabai_rawit_merah']
SEMUA, TANPA_MODEL, HANYA_MODEL = [0, 1, 2, 3, 4, 5], [0, 1, 2, 3], [4, 5]


def latih(a, kolom, benih=0):
    m = a['ok']
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.8, random_state=benih, verbose=-1)
    kat = [kolom.index(2)] if 2 in kolom else 'auto'
    clf.fit(a['X'][m][:, kolom], a['lab'][m], categorical_feature=kat)
    return clf


def peluang(clf, b, kolom):
    p = np.full(b['ok'].shape, np.nan)
    p[b['ok']] = clf.predict_proba(b['X'][b['ok']][:, kolom])[:, 1]
    return np.nan_to_num(p)


def beku(a, b):
    """Pembanding beku dari periode latih: kelengketan dan daftar, cara yang sama dengan 16_gerbang.py."""
    ea = lambda x: np.where(a['ok'], np.abs(a['y'] - x), np.nan) / a['skala']
    ma, na_ = ea(a['q'][:, :, 1]), ea(a['c'])
    gz = np.arange(0.2, 1.01, 0.1)
    tz = gz[int(np.nanargmin([np.nanmean(np.where(a['X'][:, :, 0] < g, ma, na_)) for g in gz]))]
    lis = np.array([np.nanmean(ma[kom == kk]) < np.nanmean(na_[kom == kk]) for kk in kom])
    f1 = np.where(b['X'][:, :, 0] < tz, b['q'][:, :, 1], b['c'])
    f2 = np.where(lis[:, None], b['q'][:, :, 1], b['c'])
    return f1, f2, tz, sorted(set(kom[lis]))


def bedakan(x, b, acuan, idx):
    ok, y, sk = b['ok'], b['y'], b['skala']
    e = lambda f: np.where(ok, np.abs(y - f), np.nan)
    d = e(x) - e(acuan)
    return f'MASE {G.ci(np.nanmean(d / sk, 0), idx)} MAPE {G.ci(np.nanmean(d / y * 100, 0), idx)}'


def potong(d, k):
    return {kk: v[:, k] for kk, v in d.items()}


def uji_awal():
    pra = pd.read_csv(os.path.join(T, 'asal_pra.csv')).indeks.to_numpy()
    batas = np.datetime64('2022-01-01')
    print('\n===== (A) periode lebih awal: latih sasaran < 2022, uji asal >= 2022')
    for m in ('timesfm', 'moirai'):
        Q = np.load(os.path.join(T, f'pred_{m}_pra.npy'))
        for h in HORIZON:
            d = G.siapkan(pra, Q, h)
            ta, ts = tgl.to_numpy()[pra], tgl.to_numpy()[pra + h]
            kl, ku = np.flatnonzero(ts < batas), np.flatnonzero(ta >= batas)
            a, b = potong(d, kl), potong(d, ku)
            b['ok'] = b['ok'] & np.isfinite(b['q']).all(2)
            p = peluang(latih(a, SEMUA), b, SEMUA)
            gm = G.median_campuran(p, b['c'], b['q'])
            f1, f2, tz, lis = beku(a, b)
            idx = G.boot(len(ku), 20261500 + h)
            print(f'{m} h={h}: latih {len(kl)} asal, uji {len(ku)} asal, ambang {tz:.1f}; daftar {lis}')
            print(f'  AUC {G.auc(p[b["ok"]], b["lab"][b["ok"]]):.3f}')
            print('  model lawan naif:   ' + bedakan(b['q'][:, :, 1], b, b['c'], idx))
            print('  gerbang lawan naif: ' + bedakan(gm, b, b['c'], idx))
            print('  gerbang lawan model:' + bedakan(gm, b, b['q'][:, :, 1], idx))
            print('  kelengketan lawan naif: ' + bedakan(f1, b, b['c'], idx))
            print('  daftar lawan naif:  ' + bedakan(f2, b, b['c'], idx))
            print('  gerbang lawan kelengketan: ' + bedakan(gm, b, f1, idx))
            print('  gerbang lawan daftar: ' + bedakan(gm, b, f2, idx), flush=True)


def uji_utama():
    pra = pd.read_csv(os.path.join(T, 'asal_pra.csv')).indeks.to_numpy()
    uji = pd.read_csv(os.path.join(T, 'asal.csv')).indeks.to_numpy()
    Qp, Qu = np.load(os.path.join(T, 'pred_timesfm_pra.npy')), np.load(os.path.join(T, 'pred_timesfm.npy'))
    ft = {n: np.load(os.path.join(T, f'pred_chronos2{s}.npy')) for n, s in (('dilatih', '_ft'), ('dilatih+kov', '_ftkov'))}
    for h in HORIZON:
        a, b = G.siapkan(pra, Qp, h), G.siapkan(uji, Qu, h)
        idx = G.boot(len(uji), 20261600 + h)
        ok, c, q = b['ok'], b['c'], b['q']
        p = peluang(latih(a, SEMUA), b, SEMUA)
        gm = G.median_campuran(p, c, q)
        print(f'\n===== h={h}')
        print('(B) ablasi')
        for nama, kol in (('tanpa ciri model', TANPA_MODEL), ('hanya ciri model', HANYA_MODEL)):
            pk = peluang(latih(a, kol), b, kol)
            x = G.median_campuran(pk, c, q)
            print(f'  {nama}: AUC {G.auc(pk[ok], b["lab"][ok]):.3f}; lawan naif {bedakan(x, b, c, idx)}; penuh dikurangi ini {bedakan(gm, b, x, idx)}')
        print('(C) kalibrasi per desil p (rerata p / frekuensi diam)')
        s = pd.DataFrame({'p': p[ok], 'l': b['lab'][ok]})
        s['d'] = pd.qcut(s.p, 10, labels=False, duplicates='drop')
        print('  ' + ' '.join(f'{r.p:.2f}/{r.l:.2f}' for r in s.groupby('d').mean().itertuples()))
        print('(D) lawan Chronos-2 dilatih')
        for n, Q in ft.items():
            print(f'  gerbang lawan {n}: ' + bedakan(gm, b, Q[:, :, h - 1, 1].astype(float), idx))
        print('(E) kelompok')
        d = np.where(ok, np.abs(b['y'] - gm) - np.abs(b['y'] - c), np.nan)
        dt = np.where(ok, np.abs(b['y'] - gm) - np.abs(b['y'] - q[:, :, 1]), np.nan)
        for nama, s_ in (('bergerak', BERGERAK), ('lengket', G.LENGKET)):
            r = np.isin(kom, s_)
            print(f'  {nama}: tepat di harga asal {np.mean((gm == c)[r][ok[r]]) * 100:.1f} %; lawan naif MAPE '
                  f'{G.ci(np.nanmean((d / b["y"] * 100)[r], 0), idx)} MASE {G.ci(np.nanmean((d / b["skala"])[r], 0), idx)}; '
                  f'lawan TimesFM MAPE {G.ci(np.nanmean((dt / b["y"] * 100)[r], 0), idx)}')
        print(f'  semua: tepat di harga asal {np.mean((gm == c)[ok]) * 100:.1f} %')
        print('(F) lima benih, lawan naif')
        ms, ap = [], []
        for bn in range(5):
            x = G.median_campuran(peluang(latih(a, SEMUA, bn), b, SEMUA), c, q)
            e = np.where(ok, np.abs(b['y'] - x) - np.abs(b['y'] - c), np.nan)
            ms.append(np.nanmean(np.nanmean(e / b['skala'], 0)))
            ap.append(np.nanmean(np.nanmean(e / b['y'] * 100, 0)))
        print(f'  MASE {min(ms):+.3f} s.d. {max(ms):+.3f}; MAPE {min(ap):+.3f} s.d. {max(ap):+.3f}', flush=True)


if __name__ == '__main__':
    uji_awal()
    uji_utama()
