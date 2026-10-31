"""PIHPS — gerbang diam-atau-gerak di atas model fondasi zero-shot, dan kalibrasi konformal intervalnya.

Rancangan ditulis di Hasil_Uji_PIHPS.md ("Rancangan metode") sebelum skrip ini dijalankan.
(1) LightGBM memprakirakan p = peluang harga pada asal + h sama dengan harga asal; dilatih pada 202 asal pra-2024,
dibekukan, diuji pada 142 asal 2024–2026. Gabungan: keras, rerata, median campuran.
(2) CQR per komoditas dan horizon dari skor pra-2024, diterapkan ke 2024–2026.

Pakai: venv-gpu python 16_gerbang.py > gerbang.log
"""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

T = os.path.join(os.environ.get('PIHPS_DATA', 'C:/Data/PIHPS'), 'turunan')
KONTEKS, HORIZON, BOOT, BLOK, JENDELA = 512, [7, 14, 28], 10000, 8, 60
SYAWAL = pd.to_datetime(['2019-06-05', '2020-05-24', '2021-05-13', '2022-05-02', '2023-04-22',
                         '2024-04-10', '2025-03-31', '2026-03-21', '2027-03-10'])
LENGKET = ['beras_medium_1', 'gula_pasir_lokal', 'minyak_goreng_curah']

Y = np.load(os.path.join(T, 'Y.npy'))
A = np.load(os.path.join(T, 'asli.npy'))
tgl = pd.to_datetime(pd.read_csv(os.path.join(T, 'tanggal.csv')).tanggal)
kom = pd.read_csv(os.path.join(T, 'deret.csv')).komoditas.to_numpy()
kode = pd.factorize(kom)[0]
survei = ~tgl.isin(pd.to_datetime(pd.read_csv(os.path.join(T, 'tanpa_survei.csv')).tanggal)).to_numpy()
n = Y.shape[0]
hs = np.flatnonzero(survei)


def ciri(asal):
    z, v = np.full((n, len(asal)), np.nan), np.full((n, len(asal)), np.nan)
    for k, o in enumerate(asal):
        j = hs[hs <= o][-JENDELA - 1:]
        d = np.diff(Y[:, j], axis=1)
        z[:, k] = np.nanmean(np.where(np.isfinite(d), d == 0, np.nan), axis=1)
        v[:, k] = np.nanmean(np.abs(np.diff(np.log(Y[:, j]), axis=1)), axis=1)
    return z, v


def siapkan(asal, Q, h):
    """Masukan, label, dan besaran penilaian untuk satu periode dan horizon."""
    z, v = ciri(asal)
    c = Y[:, asal]
    y = Y[:, asal + h]
    q = Q[:, :, h - 1, :].astype(float)
    q = np.sort(q, axis=2)
    ts = tgl.to_numpy()[asal + h]
    selisih = (ts[:, None] - SYAWAL.to_numpy()[None, :]).astype('timedelta64[D]').astype(int)
    jarak = selisih[np.arange(len(ts)), np.abs(selisih).argmin(axis=1)].astype(float)
    jarak = np.clip(jarak, -60, 60)
    skala = np.array([[np.nanmean(np.abs(np.diff(Y[i, o - KONTEKS + 1:o + 1]))) for o in asal] for i in range(n)])
    ok = np.isfinite(y) & (skala > 0) & survei[asal + h][None, :] & A[:, asal + h]
    X = np.stack([z, v, np.repeat(kode[:, None], len(asal), 1), np.repeat(jarak[None, :], n, 0),
                  (q[:, :, 2] - q[:, :, 0]) / c, (q[:, :, 1] - c) / c], axis=-1)
    return dict(X=X, lab=(y == c).astype(float), ok=ok, c=c, y=y, q=q, skala=skala)


def median_campuran(p, c, q):
    """Median dari campuran: massa p di c, sisanya distribusi linear sepotong lewat kuantil 0,1/0,5/0,9."""
    lv = np.array([0, .1, .5, .9, 1])
    out = np.empty_like(c)
    for i in np.ndindex(c.shape):
        q1, q5, q9 = q[i]
        x = np.array([q1 - (q5 - q1) / 4, q1, q5, q9, q9 + (q9 - q5) / 4])
        x = np.maximum.accumulate(x)
        F = lambda u: np.interp(u, x, lv)
        Finv = lambda a: np.interp(a, lv, x)
        bawah = (1 - p[i]) * F(c[i])
        if bawah > 0.5:
            out[i] = Finv(0.5 / (1 - p[i]))
        elif bawah + p[i] < 0.5:
            out[i] = Finv((0.5 - p[i]) / (1 - p[i]))
        else:
            out[i] = c[i]
    return out


def boot(ban, seed):
    rng = np.random.default_rng(seed)
    aw = rng.integers(0, ban - BLOK + 1, (BOOT, int(np.ceil(ban / BLOK))))
    return (aw[:, :, None] + np.arange(BLOK)).reshape(BOOT, -1)[:, :ban]


def ci(x, idx):
    b = np.nanmean(x[idx], axis=1)
    return f'{np.nanmean(x):+.3f} [{np.percentile(b, 2.5):+.3f}; {np.percentile(b, 97.5):+.3f}]'


def auc(s, l):
    r = pd.Series(s).rank().to_numpy()
    n1 = l.sum()
    return (r[l == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(l) - n1))


def main():
    pra = pd.read_csv(os.path.join(T, 'asal_pra.csv')).indeks.to_numpy()
    uji = pd.read_csv(os.path.join(T, 'asal.csv')).indeks.to_numpy()
    for m in ('timesfm', 'moirai'):
        Qp = np.load(os.path.join(T, f'pred_{m}_pra.npy'))
        Qu = np.load(os.path.join(T, f'pred_{m}.npy'))
        print(f'\n===== kandidat {m}')
        for h in HORIZON:
            a, b = siapkan(pra, Qp, h), siapkan(uji, Qu, h)
            Xa, la = a['X'][a['ok']], a['lab'][a['ok']]
            clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
            clf.fit(Xa, la, categorical_feature=[2])
            p = np.full(b['ok'].shape, np.nan)
            p[b['ok']] = clf.predict_proba(b['X'][b['ok']])[:, 1]
            lb = b['lab'][b['ok']]
            print(f'h={h}: diam pra {la.mean():.3f}, uji {lb.mean():.3f}; AUC uji {auc(p[b["ok"]], lb):.3f}; '
                  f'Brier uji {np.mean((p[b["ok"]] - lb) ** 2):.4f} (dasar {np.mean((la.mean() - lb) ** 2):.4f})')
            c, y, q, sk, ok = b['c'], b['y'], b['q'], b['skala'], b['ok']
            pp = np.nan_to_num(p)
            f = {'TimesFM' if m == 'timesfm' else 'Moirai-2': q[:, :, 1],
                 'ditahan Rp 50': np.where(np.abs(q[:, :, 1] - c) < 50, c, np.round(q[:, :, 1] / 50) * 50),
                 'gerbang keras': np.where(pp > 0.5, c, q[:, :, 1]),
                 'gerbang rerata': pp * c + (1 - pp) * q[:, :, 1],
                 'gerbang median': median_campuran(pp, c, q)}
            # pembanding beku dari pra-2024 (seperti 14_uji_pra2024.py)
            ea = lambda x: np.where(a['ok'], np.abs(a['y'] - x), np.nan) / a['skala']
            ma, na_ = ea(a['q'][:, :, 1]), ea(a['c'])
            gz = np.arange(0.2, 1.01, 0.1)
            tz = gz[int(np.nanargmin([np.nanmean(np.where(a['X'][:, :, 0] < g, ma, na_)) for g in gz]))]
            lis = np.array([np.nanmean(ma[kom == kk]) < np.nanmean(na_[kom == kk]) for kk in kom])
            f['kelengketan beku'] = np.where(b['X'][:, :, 0] < tz, q[:, :, 1], c)
            f['daftar beku'] = np.where(lis[:, None], q[:, :, 1], c)
            e = lambda x: np.where(ok, np.abs(y - x), np.nan)
            en = e(c)
            idx = boot(len(uji), 20261400 + h)
            nm = list(f)[0]
            for nama, x in f.items():
                dm, da = (e(x) - en) / sk, (e(x) - en) / y * 100
                tm, ta = (e(x) - e(f[nm])) / sk, (e(x) - e(f[nm])) / y * 100
                lk = np.isin(kom, LENGKET)
                print(f'  {nama:15s} lawan naif MASE {ci(np.nanmean(dm, 0), idx)} MAPE {ci(np.nanmean(da, 0), idx)}'
                      + ('' if nama == nm else f' | lawan {nm} MASE {ci(np.nanmean(tm, 0), idx)} MAPE {ci(np.nanmean(ta, 0), idx)}')
                      + f' | lengket MAPE {np.nanmean(da[lk]):+.3f}')
            gm = e(f['gerbang median'])
            for nama in ('ditahan Rp 50', 'kelengketan beku', 'daftar beku'):
                print(f'  gerbang median - {nama}: MASE {ci(np.nanmean((gm - e(f[nama])) / sk, 0), idx)} MAPE {ci(np.nanmean((gm - e(f[nama])) / y * 100, 0), idx)}')
            imp = clf.booster_.feature_importance('gain')
            print('  kepentingan (gain %): ' + ' '.join(f'{k}={v / imp.sum() * 100:.0f}' for k, v in
                                                       zip(['kelengketan', 'volatilitas', 'komoditas', 'Idulfitri', 'lebar', 'geser'], imp)))

            # (2) CQR per komoditas
            sa = np.maximum(a['q'][:, :, 0] - a['y'], a['y'] - a['q'][:, :, 2]) / a['c']
            lo, hi = q[:, :, 0].copy(), q[:, :, 2].copy()
            for kk in np.unique(kom):
                r = kom == kk
                s = sa[r][a['ok'][r]]
                s = s[np.isfinite(s)]
                t = np.quantile(s, min(1, np.ceil((len(s) + 1) * 0.8) / len(s)))
                lo[r] -= t * c[r]
                hi[r] += t * c[r]
            for nama, (l, u) in (('asli', (q[:, :, 0], q[:, :, 2])), ('CQR', (lo, hi))):
                cov = ((y >= l) & (y <= u))[ok]
                lebar = ((u - l) / c)[ok]
                isk = ((u - l) + 2 / 0.2 * (np.maximum(l - y, 0) + np.maximum(y - u, 0))) / c * 100
                per = ' '.join(f'{kk[:6]}={((y >= l) & (y <= u))[ok & (kom == kk)[:, None]].mean():.2f}' for kk in np.unique(kom))
                print(f'  interval {nama}: cakupan {cov.mean():.3f}, lebar rel {np.median(lebar) * 100:.2f} %, skor interval {np.nanmean(isk[ok]):.2f}; {per}')


if __name__ == '__main__':
    main()
