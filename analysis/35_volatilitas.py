"""PIHPS — apakah keunggulan TimesFM atas momentum hanya efek volatilitas atau provinsi? Klimatologi per deret, momentum + volatilitas,
AUPRC di dalam deret, jumlah episode. h 14, ambang 20 %.

Rancangan ditulis dan di-commit sebelum dijalankan: Hasil_Uji_PIHPS.md, "Rancangan skrip 35" (commit 8e1f555).
Semua bagian lain (kejadian, fitur, latih, isotonik, bootstrap) diambil dari 33_peringatan.py tanpa perubahan.

Pakai: venv-gpu python 35_volatilitas.py > volatilitas.log
       PIHPS_DATA=C:/Data/PIHPS_lain venv-gpu python 35_volatilitas.py > volatilitas_lain.log
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

D = os.path.dirname(os.path.abspath(__file__))
s_ = importlib.util.spec_from_file_location('p33', os.path.join(D, '33_peringatan.py'))
P = importlib.util.module_from_spec(s_)
s_.loader.exec_module(P)
G, T, Y, LY, n, kom = P.G, P.T, P.Y, P.LY, P.n, P.kom
H, AMB = 14, 0.20


def volatil(asal):
    """log simpangan baku perubahan log harian 60 dan 250 hari terakhir, hanya dari Y[..o]."""
    out = []
    for k in (60, 250):
        d = np.stack([np.diff(LY[:, o - k:o + 1], axis=1) for o in asal], axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            out.append(np.log(np.nanstd(d, 2) + 1e-4))
    return np.stack(out, -1)


def ap_dalam(skor, l, sr, w=None):
    """Rerata AUPRC per deret berbobot jumlah kejadian (berbobot w); deret dengan kejadian dan bukan-kejadian saja."""
    w = np.ones(len(l)) if w is None else w
    tot, bobot = 0.0, 0.0
    for idx in sr:
        ww, ll = w[idx], l[idx]
        pos = np.sum(ww * ll)
        if pos <= 0 or np.sum(ww * (1 - ll)) <= 0:
            continue
        v = skor[idx]
        o = np.argsort(-v, kind='stable')
        tot += pos * P.ap(v[o], ll[o], ww[o])
        bobot += pos
    return tot / bobot


def episode(e, ok):
    """e, ok: (deret, asal). Episode per deret = rangkaian asal berurutan dengan kejadian."""
    x = (e == 1) & ok
    awal = x & ~np.concatenate([np.zeros((x.shape[0], 1), bool), x[:, :-1]], 1)
    return int(awal.sum()), awal


def main():
    uji = pd.read_csv(os.path.join(T, 'asal.csv')).indeks.to_numpy()
    pra = pd.read_csv(os.path.join(T, 'asal_pra.csv')).indeks.to_numpy()
    semua = np.arange(P.KONTEKS, P.BATAS - H)
    Xs, Xu = P.ciri(semua), P.ciri(uji)
    Vs, Vu = volatil(semua), volatil(uji)
    oks, oku = P.okmask(semua, H), P.okmask(uji, H)
    pp = pra[pra + H < P.BATAS]
    jp = np.searchsorted(pra, pp)
    okp = P.okmask(pp, H)
    l_s, l_p, l_u = P.kejadian(semua, H, AMB)[oks], P.kejadian(pp, H, AMB)[okp], P.kejadian(uji, H, AMB)[oku]
    kf = pd.factorize(kom)[0]
    kk_s, kk_u = np.repeat(kf[:, None], len(semua), 1)[oks], np.repeat(kf[:, None], len(uji), 1)[oku]
    dr_s, dr_u = np.repeat(np.arange(n)[:, None], len(semua), 1)[oks], np.repeat(np.arange(n)[:, None], len(uji), 1)[oku]
    nk = kf.max() + 1

    skor = {}
    klim = pd.Series(l_s).groupby(kk_s).mean()
    skor['klimatologi'] = klim.reindex(kk_u).to_numpy()
    kd = pd.Series(l_s).groupby(dr_s).mean().reindex(np.arange(n))
    kd = kd.fillna(pd.Series(klim.reindex(kf).to_numpy())).to_numpy()
    skor['klimatologi per deret'] = kd[dr_u]

    def mom(X, kk):
        return np.nan_to_num(np.column_stack([np.eye(nk)[kk], X[:, 4], X[:, 19], X[:, 22]]))

    def logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    def momv(X, V, kk, dr):
        return np.nan_to_num(np.column_stack([mom(X, kk), V, logit(kd[dr])]))

    lr = LogisticRegression(max_iter=2000).fit(mom(Xs[oks], kk_s), l_s)
    skor['momentum'] = lr.predict_proba(mom(Xu[oku], kk_u))[:, 1]
    lr2 = LogisticRegression(max_iter=4000).fit(momv(Xs[oks], Vs[oks], kk_s, dr_s), l_s)
    skor['momentum + volatilitas'] = lr2.predict_proba(momv(Xu[oku], Vu[oku], kk_u, dr_u))[:, 1]
    import lightgbm as lgb
    gb = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=P.BENIH, verbose=-1).fit(
        Xs[oks], l_s, categorical_feature=[24, 25])
    skor['LightGBM'] = gb.predict_proba(Xu[oku])[:, 1]
    Qp = np.load(os.path.join(T, 'pred_timesfm_pra_d9.npy'))[:, jp, H - 1, :].astype(float)
    Qu = np.load(os.path.join(T, 'pred_timesfm_d9.npy'))[:, :, H - 1, :].astype(float)
    ap_, au_ = (1 + AMB) * Y[:, pp], (1 + AMB) * Y[:, uji]
    rp, ru = P.peluang(Qp, ap_), P.peluang(Qu, au_)
    tu_ = 1e-6 * Qu[..., 8] / au_
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(rp[okp], l_p)
    skor['TimesFM 2.5 mentah'] = ru[oku] + tu_[oku]
    skor['TimesFM 2.5 terkalibrasi'] = iso.predict(ru[oku]) + tu_[oku]

    print(f'h={H}, ambang {AMB:.0%}: pasangan uji {len(l_u)}, kejadian {int(l_u.sum())} ({l_u.mean():.3f}); latih {len(l_s)}')
    e = P.kejadian(uji, H, AMB)
    ne, awal = episode(e, P.okmask(uji, H))
    print(f'episode per deret (asal uji berurutan dengan kejadian): {ne}; deret dengan kejadian: {int(((e == 1) & P.okmask(uji, H)).any(1).sum())}')
    gab = 0
    for j in range(nk):
        t = np.sort(np.flatnonzero(awal[kf == j].any(0)))
        if len(t):
            gab += 1 + int(np.sum(np.diff(t) > 2))
    print(f'episode lintas provinsi per komoditas (awal berselang <= 2 asal digabung): {gab}')

    sr = [np.flatnonzero(dr_u == d) for d in range(n)]
    sr = [i for i in sr if len(i)]
    print(f'deret yang masuk AUPRC di dalam deret: {sum(1 for i in sr if 0 < l_u[i].sum() < len(i))}')
    kol = np.repeat(np.arange(len(uji))[None, :], n, 0)[oku]
    G.BOOT = P.NBOOT
    idx = G.boot(len(uji), P.BENIH + H)
    urut = {k: np.argsort(-v, kind='stable') for k, v in skor.items()}
    bg = {k: np.empty(len(idx)) for k in skor}
    bd = {k: np.empty(len(idx)) for k in skor}
    for b in range(len(idx)):
        w = np.bincount(idx[b], minlength=kol.max() + 1)[kol].astype(float)
        for k, v in skor.items():
            o = urut[k]
            bg[k][b] = P.ap(v[o], l_u[o], w[o])
            bd[k][b] = ap_dalam(v, l_u, sr, w)
    tg, td = {}, {}
    print('  metode: AUPRC gabungan [CI95] | AUPRC di dalam deret [CI95]')
    for k, v in skor.items():
        o = urut[k]
        tg[k], td[k] = P.ap(v[o], l_u[o], np.ones(len(v))), ap_dalam(v, l_u, sr)
        print(f'    {k:26s} {P.teks(tg[k], bg[k])} | {P.teks(td[k], bd[k])}')
    print('  selisih dengan CI95 (indeks bootstrap sama): gabungan | di dalam deret')
    hasil = {}
    for y in ('momentum + volatilitas', 'klimatologi per deret', 'momentum', 'LightGBM', 'klimatologi'):
        for x in ('TimesFM 2.5 terkalibrasi', 'TimesFM 2.5 mentah'):
            a, b2 = bg[x] - bg[y], bd[x] - bd[y]
            hasil[(x, y)] = (np.percentile(a, 2.5), np.percentile(b2, 2.5))
            print(f'    {x} − {y}: {P.teks(tg[x] - tg[y], a)} | {P.teks(td[x] - td[y], b2)}')
    x = 'TimesFM 2.5 terkalibrasi'
    print(f'\n[V1] gabungan − momentum + volatilitas: batas bawah {hasil[(x, "momentum + volatilitas")][0]:+.3f} '
          f'{"LULUS" if hasil[(x, "momentum + volatilitas")][0] > 0 else "GAGAL"}')
    print(f'[V2] gabungan − klimatologi per deret: batas bawah {hasil[(x, "klimatologi per deret")][0]:+.3f} '
          f'{"LULUS" if hasil[(x, "klimatologi per deret")][0] > 0 else "GAGAL"}')
    print(f'[V3] di dalam deret − momentum + volatilitas: batas bawah {hasil[(x, "momentum + volatilitas")][1]:+.3f} '
          f'{"LULUS" if hasil[(x, "momentum + volatilitas")][1] > 0 else "GAGAL"}')


if __name__ == '__main__':
    main()
