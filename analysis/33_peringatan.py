"""PIHPS — peringatan dini lonjakan harga dari kuantil model fondasi zero-shot, lawan klimatologi, momentum, dan LightGBM.

Rancangan ditulis dan di-commit sebelum dijalankan: Hasil_Uji_PIHPS.md, "Rancangan skrip 33" (commit a413d80).
Kejadian: Y[o+h] >= (1 + ambang) * Y[o]; utama h 14, ambang 20 %. Uji: 142 asal 2024–2026; latih: sasaran < indeks 1548.
Butuh prakiraan desil: PIHPS_KUANTIL=9 python 03_model_fondasi.py <model> (dan PIHPS_ASAL=pra).

Pakai: venv-gpu python 33_peringatan.py > peringatan.log
"""
import importlib.util
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

D = os.path.dirname(os.path.abspath(__file__))
s_ = importlib.util.spec_from_file_location('g', os.path.join(D, '16_gerbang.py'))
G = importlib.util.module_from_spec(s_)
s_.loader.exec_module(G)
T, Y, A, survei, tgl, kom = G.T, G.Y.astype(float), G.A, G.survei, G.tgl, G.kom
wil = pd.read_csv(os.path.join(T, 'deret.csv')).wilayah.to_numpy()
BATAS, KONTEKS, NBOOT, BENIH = 1548, 512, 2000, 20263400
HORIZON, AMBANG = [14, 7, 28], [0.20, 0.10]
MODEL = [('Chronos-2', 'chronos2'), ('Chronos-Bolt', 'chronosbolt'), ('TimesFM 2.5', 'timesfm'), ('Moirai-2', 'moirai')]
LV = np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
LY = np.log(Y)
n = Y.shape[0]


def okmask(asal, h):
    return np.isfinite(Y[:, asal]) & np.isfinite(Y[:, asal + h]) & survei[asal + h][None, :] & A[:, asal + h]


def kejadian(asal, h, amb):
    with np.errstate(invalid='ignore'):
        return (Y[:, asal + h] >= (1 + amb) * Y[:, asal]).astype(int)


def ciri(asal):
    """Fitur di asal o, hanya dari Y[..o]. Keluaran (n, asal, fitur)."""
    f = [LY[:, asal] - LY[:, asal - k] for k in list(range(1, 21)) + [40, 60]]
    w = np.stack([LY[:, o - 89:o + 1] for o in asal], axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        z = (LY[:, asal] - np.nanmean(w, 2)) / np.nanstd(w, 2)
    d = np.stack([np.diff(Y[:, o - 60:o + 1], axis=1) for o in asal], axis=1)
    diam = np.nanmean(np.where(np.isfinite(d), d == 0, np.nan), 2)
    m = len(asal)
    kk = np.repeat(pd.factorize(kom)[0][:, None], m, 1)
    ww = np.repeat(pd.factorize(wil)[0][:, None], m, 1)
    bl = np.repeat(tgl.dt.month.to_numpy()[asal][None, :], n, 0)
    hr = np.repeat(tgl.dt.dayofyear.to_numpy()[asal][None, :], n, 0)
    return np.stack(f + [z, diam, kk, ww, bl, hr], axis=-1).astype(float)


def peluang(q, ambang):
    """P(Y >= ambang) dari CDF linear sepotong pada kuantil 0,1..0,9, ekor diperpanjang linear."""
    q = np.sort(q, axis=-1)
    x = np.concatenate([q[..., :1] - (q[..., 1:2] - q[..., :1]), q, q[..., -1:] + (q[..., -1:] - q[..., -2:-1])], -1)
    x = np.maximum.accumulate(x, axis=-1)
    F = np.zeros(ambang.shape)
    for j in range(10):
        a, b = x[..., j], x[..., j + 1]
        with np.errstate(invalid='ignore', divide='ignore'):
            fr = np.where(b > a, np.clip((ambang - a) / (b - a), 0, 1), (ambang >= a).astype(float))
        F += fr * (LV[j + 1] - LV[j])
    return np.clip(1 - F, 0, 1)


def ap(s, l, w):
    """Presisi rata-rata berbobot (sama dengan sklearn average_precision_score); s sudah diurutkan menurun."""
    tp, fp = np.cumsum(w * l), np.cumsum(w * (1 - l))
    ujung = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp, fp = tp[ujung], fp[ujung]
    pr = tp / np.maximum(tp + fp, 1e-12)
    rc = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0, rc]) * pr))


def ambang_f1(s, l):
    o = np.argsort(-s)
    tp = np.cumsum(l[o])
    pr, rc = tp / np.arange(1, len(s) + 1), tp / l.sum()
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    return s[o][np.argmax(f1)]


def bootstrap(skor, l, kol, idx):
    """AUPRC dan BS per ulangan untuk tiap metode, indeks asal sama."""
    urut = {k: np.argsort(-v, kind='stable') for k, v in skor.items()}
    hasil = {k: np.empty((len(idx), 2)) for k in skor}
    for b in range(len(idx)):
        w = np.bincount(idx[b], minlength=kol.max() + 1)[kol].astype(float)
        for k, v in skor.items():
            o = urut[k]
            hasil[k][b] = ap(v[o], l[o], w[o]), np.sum(w * (v - l) ** 2) / w.sum()
    return hasil


def teks(x, b):
    return f'{x:+.3f} [{np.percentile(b, 2.5):+.3f}; {np.percentile(b, 97.5):+.3f}]'


def main():
    uji = pd.read_csv(os.path.join(T, 'asal.csv')).indeks.to_numpy()
    pra = pd.read_csv(os.path.join(T, 'asal_pra.csv')).indeks.to_numpy()
    Qu = {s: np.load(os.path.join(T, f'pred_{s}_d9.npy')) for _, s in MODEL}
    Qp = {s: np.load(os.path.join(T, f'pred_{s}_pra_d9.npy')) for _, s in MODEL}
    for _, s in MODEL:
        lama = np.load(os.path.join(T, f'pred_{s}.npy'))[..., 1]
        print(f'periksa median d9 lawan pred lama {s}: selisih maksimum {np.nanmax(np.abs(Qu[s][..., 4] - lama)):.4f}, '
              f'relatif {np.nanmax(np.abs(Qu[s][..., 4] - lama) / lama):.2e}')
    Xu = ciri(uji)
    G.BOOT = NBOOT
    keputusan = None
    for h in HORIZON:
        semua = np.arange(KONTEKS, BATAS - h)
        Xs, oks = ciri(semua), okmask(semua, h)
        pp = pra[pra + h < BATAS]
        jp = np.searchsorted(pra, pp)
        okp, oku = okmask(pp, h), okmask(uji, h)
        idx = G.boot(len(uji), BENIH + h)
        kol = np.repeat(np.arange(len(uji))[None, :], n, 0)[oku]
        for amb in AMBANG:
            l_s, l_p, l_u = kejadian(semua, h, amb)[oks], kejadian(pp, h, amb)[okp], kejadian(uji, h, amb)[oku]
            kk_s, kk_u = np.repeat(pd.factorize(kom)[0][:, None], len(semua), 1)[oks], np.repeat(pd.factorize(kom)[0][:, None], len(uji), 1)[oku]
            klim = pd.Series(l_s).groupby(kk_s).mean()
            skor, latih = {}, {}
            skor['klimatologi'] = klim.reindex(kk_u).to_numpy()
            latih['klimatologi'] = klim.reindex(kk_s).to_numpy(), l_s
            # momentum: regresi logistik, boneka komoditas + perubahan 5 dan 20 hari + z-score 90 hari
            def mom(X, kk):
                b = np.eye(len(klim))[kk]
                return np.nan_to_num(np.column_stack([b, X[:, 4], X[:, 19], X[:, 22]]))
            lr = LogisticRegression(max_iter=2000).fit(mom(Xs[oks], kk_s), l_s)
            skor['momentum'] = lr.predict_proba(mom(Xu[oku], kk_u))[:, 1]
            latih['momentum'] = lr.predict_proba(mom(Xs[oks], kk_s))[:, 1], l_s
            par = dict(n_estimators=500, learning_rate=0.05, random_state=BENIH, verbose=-1)
            kat = [24, 25]
            gb = lgb.LGBMClassifier(**par).fit(Xs[oks], l_s, categorical_feature=kat)
            skor['LightGBM'] = gb.predict_proba(Xu[oku])[:, 1]
            latih['LightGBM'] = gb.predict_proba(Xs[oks])[:, 1], l_s
            mentah_p, mentah_u = {}, {}
            for nama, s in MODEL:
                qp, qu = Qp[s][:, jp, h - 1, :].astype(float), Qu[s][:, :, h - 1, :].astype(float)
                ap_, au_ = (1 + amb) * Y[:, pp], (1 + amb) * Y[:, uji]
                rp, ru = peluang(qp, ap_), peluang(qu, au_)
                tp_, tu_ = 1e-6 * qp[..., 8] / ap_, 1e-6 * qu[..., 8] / au_
                mentah_p[s], mentah_u[s] = rp[okp], ru[oku]
                iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(rp[okp], l_p)
                skor[f'{nama} mentah'] = ru[oku] + tu_[oku]
                latih[f'{nama} mentah'] = rp[okp] + tp_[okp], l_p
                skor[f'{nama} terkalibrasi'] = iso.predict(ru[oku]) + tu_[oku]
                latih[f'{nama} terkalibrasi'] = iso.predict(rp[okp]) + tp_[okp], l_p
            # LightGBM + skor mentah TimesFM, latih pada asal_pra saja
            Xp = ciri(pp)[okp]
            gb2 = lgb.LGBMClassifier(**par).fit(np.column_stack([Xp, mentah_p['timesfm']]), l_p, categorical_feature=kat)
            skor['LightGBM + TimesFM'] = gb2.predict_proba(np.column_stack([Xu[oku], mentah_u['timesfm']]))[:, 1]
            latih['LightGBM + TimesFM'] = gb2.predict_proba(np.column_stack([Xp, mentah_p['timesfm']]))[:, 1], l_p
            gb3 = lgb.LGBMClassifier(**par).fit(Xp, l_p, categorical_feature=kat)
            skor['LightGBM (latih pra)'] = gb3.predict_proba(Xu[oku])[:, 1]
            latih['LightGBM (latih pra)'] = gb3.predict_proba(Xp)[:, 1], l_p

            print(f'\n===== h={h}, ambang {amb:.0%}: pasangan uji {len(l_u)}, kejadian {l_u.mean():.3f}; '
                  f'latih {len(l_s)} ({l_s.mean():.3f}), pra {len(l_p)} ({l_p.mean():.3f})', flush=True)
            bs = bootstrap(skor, l_u, kol, idx)
            bs_k = np.mean((skor['klimatologi'] - l_u) ** 2)
            tit = {}
            print('  metode: AUPRC [CI95] | BSS terhadap klimatologi | pada ambang F1 latih: presisi, recall, laju alarm palsu')
            for k, v in skor.items():
                o = np.argsort(-v, kind='stable')
                a = ap(v[o], l_u[o], np.ones(len(v)))
                bss = 1 - np.mean((v - l_u) ** 2) / bs_k
                tit[k] = a, bss
                t = ambang_f1(*latih[k]) if k in latih else np.inf
                al = v >= t
                pr = (al & (l_u == 1)).sum() / max(al.sum(), 1)
                rc = (al & (l_u == 1)).sum() / l_u.sum()
                fa = (al & (l_u == 0)).sum() / (l_u == 0).sum()
                print(f'    {k:28s} {teks(a, bs[k][:, 0])} | {bss:+.3f} | {pr:.3f}, {rc:.3f}, {fa:.3f}')
            print('  selisih (AUPRC; BSS) dengan CI95, indeks bootstrap sama:')
            pas = [('TimesFM 2.5 terkalibrasi', 'LightGBM'), ('TimesFM 2.5 terkalibrasi', 'momentum'),
                   ('TimesFM 2.5 terkalibrasi', 'klimatologi'), ('LightGBM + TimesFM', 'LightGBM (latih pra)'),
                   ('LightGBM + TimesFM', 'LightGBM')]
            pas += [(f'{nm} terkalibrasi', 'LightGBM') for nm, _ in MODEL if nm != 'TimesFM 2.5']
            pas += [(f'{nm} terkalibrasi', 'momentum') for nm, _ in MODEL if nm != 'TimesFM 2.5']
            pas += [(f'{nm} terkalibrasi', f'{nm} mentah') for nm, _ in MODEL]
            beda = {}
            for x, y in pas:
                da = tit[x][0] - tit[y][0]
                bb = bs[x][:, 0] - bs[y][:, 0]
                dbss = tit[x][1] - tit[y][1]
                bb2 = (1 - bs[x][:, 1] / bs['klimatologi'][:, 1]) - (1 - bs[y][:, 1] / bs['klimatologi'][:, 1])
                beda[(x, y)] = (np.percentile(bb, 2.5), np.percentile(bb, 97.5))
                print(f'    {x} − {y}: AUPRC {teks(da, bb)}; BSS {teks(dbss, bb2)}')
            print('  AUPRC per komoditas (kejadian uji):')
            for j, kn in enumerate(pd.factorize(kom)[1]):
                r = kk_u == j
                if l_u[r].sum() < 5:
                    print(f'    {kn:22s} kejadian {int(l_u[r].sum())}: terlalu sedikit')
                    continue
                isi = []
                for k in ('klimatologi', 'momentum', 'LightGBM', 'TimesFM 2.5 terkalibrasi', 'Chronos-2 terkalibrasi', 'LightGBM + TimesFM'):
                    v = skor[k][r]
                    o = np.argsort(-v, kind='stable')
                    isi.append(f'{k} {ap(v[o], l_u[r][o], np.ones(r.sum())):.3f}')
                print(f'    {kn:22s} kejadian {int(l_u[r].sum())} ({l_u[r].mean():.3f}): ' + ', '.join(isi))
            if h == 14 and amb == 0.20:
                u1 = beda[('TimesFM 2.5 terkalibrasi', 'LightGBM')]
                u2 = beda[('LightGBM + TimesFM', 'LightGBM (latih pra)')]
                u3 = beda[('TimesFM 2.5 terkalibrasi', 'momentum')]
                if u1[0] > 0 or u2[0] > 0:
                    keputusan = 'KUAT'
                elif u1[0] <= 0 <= u1[1] and u3[0] > 0:
                    keputusan = 'SETARA'
                else:
                    keputusan = 'LEMAH'
                print(f'  [utama] TimesFM terkal − LightGBM CI {u1}; LightGBM+TimesFM − LightGBM(pra) CI {u2}; TimesFM terkal − momentum CI {u3}')
        # cakupan selang 80 % per rezim (tidak bergantung ambang; kejadian = ambang 20 %)
        e = kejadian(uji, h, 0.20)
        datar = Y[:, uji + h] == Y[:, uji]
        print(f'  cakupan selang 80 % (q0,1–q0,9) dan lebar relatif median, h={h}: datar | lonjakan 20 % | lainnya')
        for nama, s in MODEL:
            q = np.sort(Qu[s][:, :, h - 1, :].astype(float), axis=-1)
            y = Y[:, uji + h]
            masuk = (y >= q[..., 0]) & (y <= q[..., 8])
            lebar = (q[..., 8] - q[..., 0]) / Y[:, uji]
            isi = []
            for r in (datar & oku, (e == 1) & oku, ~datar & (e == 0) & oku):
                isi.append(f'{masuk[r].mean():.3f} / {np.median(lebar[r]):.3f} (n {int(r.sum())})')
            print(f'    {nama:13s} ' + ' | '.join(isi), flush=True)
    print(f'\n===== aturan keputusan (rancangan, h 14, ambang 20 %): {keputusan}')


if __name__ == '__main__':
    main()
