"""PIHPS — pemeriksaan yang diminta telaah 1 Nov 2026. Metode tidak berubah; rancangan di Hasil_Uji_PIHPS.md ("Rancangan skrip 27").

(A) pembanding baru dari data latih: p global, zona mati k langkah Rp 50, susut model-naif, rerata 0,5.
(B) rincian per kelompok dan komoditas. (C) bentuk campuran pada 2022-2023. (D) skala MASE tanggal survei dan ambang
tanggal tanpa survei, laju per tahun. (E) peringatan >= 5 % pada prakiraan campuran. (F) 5 asal yang belum pernah dinilai.
Persiapan (F): asal_baru.csv = np.arange(asal.csv terakhir + 5, jumlah tanggal - 7, 5), kolom `indeks`; lalu
PIHPS_ASAL=baru venv-gpu python 03_model_fondasi.py timesfm > fondasi_timesfm_baru.log
Pakai: venv-gpu python 27_telaah_akhir.py > telaah_akhir.log
"""
import importlib.util
import json
import os

import numpy as np
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
s_ = importlib.util.spec_from_file_location('cd', os.path.join(D, '24_campuran_dasar.py'))
C = importlib.util.module_from_spec(s_)
s_.loader.exec_module(C)
P, G, L, T, kom, dasar, HORIZON, tgl = C.P, C.G, C.L, C.T, C.kom, C.dasar, C.HORIZON, C.tgl
BERGERAK, LENGKET = C.BERGERAK, C.LENGKET
LAIN = sorted(set(kom) - set(BERGERAK) - set(LENGKET))
KOM = np.unique(kom)
GK = np.arange(0, 21)
GW = np.arange(0, 1.0001, 0.05)


def zona(m, c, k):
    return np.where(np.abs(m - c) < 50 * max(k, 1), c, np.round(m / 50) * 50) if k else m


def setel(a):
    """Pilih k dan w per komoditas dari MAE latih; juga p global."""
    ok, y, c, m = a['ok'], a['y'], a['c'], a['q'][:, :, 1]
    kk, ww = {}, {}
    for k_ in KOM:
        r = (kom == k_)[:, None] & ok & np.isfinite(c)  # 14 sel latih tanpa harga asal
        kk[k_] = int(GK[np.argmin([np.mean(np.abs(y - zona(m, c, k))[r]) for k in GK])])
        ww[k_] = float(GW[np.argmin([np.mean(np.abs(y - (w * m + (1 - w) * c))[r]) for w in GW])])
    return kk, ww, float(a['lab'][ok].mean())


def pakai(b, kk, ww, pg):
    c, m = b['c'], b['q'][:, :, 1]
    z = np.empty_like(c)
    s = np.empty_like(c)
    for k_ in KOM:
        r = kom == k_
        z[r] = zona(m[r], c[r], kk[k_])
        s[r] = ww[k_] * m[r] + (1 - ww[k_]) * c[r]
    return z, s, np.full(c.shape, pg)


def pembanding(a, b, n, seed, nama):
    pm, _ = dasar(a, n)
    x = G.median_campuran(pm, b['c'], b['q'])
    kk, ww, pg = setel(a)
    z, s, pgm = pakai(b, kk, ww, pg)
    gl = G.median_campuran(pgm, b['c'], b['q'])
    print(f'{nama}: p global {pg:.3f}; k {[kk[k] for k in KOM]}; w {[ww[k] for k in KOM]}')
    idx = G.boot(n, seed)
    for t, f in (('naif', b['c']), ('model', b['q'][:, :, 1]), ('p global', gl), ('zona mati', z), ('susut', s),
                 ('rerata 0,5', (b['c'] + b['q'][:, :, 1]) / 2)):
        print(f'    campuran dikurangi {t}: ' + P.bedakan(x, b, f, idx), flush=True)
    print('    susut dikurangi naif: ' + P.bedakan(s, b, b['c'], idx))
    print('    zona mati dikurangi naif: ' + P.bedakan(z, b, b['c'], idx))
    return x, pm


def kelompok(x, m, b, idx, label):
    ok, y, c, sk = b['ok'], b['y'], b['c'], b['skala']
    e = lambda f: np.where(ok, np.abs(y - f), np.nan)
    total = np.nansum((e(x) - e(m)) / sk)
    for nama, s in (('lengket', LENGKET), ('bergerak', BERGERAK), ('lain', LAIN)):
        r = np.isin(kom, s)
        sel = lambda f: np.nanmean(((e(f) - e(c)) / sk)[r], 0)
        pct = lambda f: np.nanmean(((e(f) - e(c)) / y * 100)[r], 0)
        g = np.nansum(((e(x) - e(m)) / sk)[r])
        print(f'  {label} {nama}: MASE saja {G.ci(sel(m), idx)} campuran {G.ci(sel(x), idx)}; MAPE saja {G.ci(pct(m), idx)} '
              f'campuran {G.ci(pct(x), idx)}; bagian keuntungan MASE {g / total * 100:.1f} %; '
              f'MAE Rp saja {np.nanmean(e(m)[r]):.0f} campuran {np.nanmean(e(x)[r]):.0f} naif {np.nanmean(e(c)[r]):.0f}', flush=True)


def skala_survei(asal, Y, survei):
    hs = np.flatnonzero(survei)
    out = np.empty((Y.shape[0], len(asal)))
    for k, o in enumerate(asal):
        j = hs[(hs > o - G.KONTEKS) & (hs <= o)]
        out[:, k] = np.nanmean(np.abs(np.diff(Y[:, j], axis=1)), axis=1)
    return out


def main():
    pra = pd.read_csv(os.path.join(T, 'asal_pra.csv')).indeks.to_numpy()
    uji = pd.read_csv(os.path.join(T, 'asal.csv')).indeks.to_numpy()
    baru = pd.read_csv(os.path.join(T, 'asal_baru.csv')).indeks.to_numpy()
    Y, A = G.Y, G.A
    batas = np.datetime64('2022-01-01')

    print('===== (A) pembanding baru, 142 asal')
    for nama, s in (('TimesFM', 'timesfm'), ('Moirai-2', 'moirai')):
        Qp, Qu = np.load(os.path.join(T, f'pred_{s}_pra.npy')), np.load(os.path.join(T, f'pred_{s}.npy'))
        for h in HORIZON:
            pembanding(G.siapkan(pra, Qp, h), G.siapkan(uji, Qu, h), len(uji), 20270100 + h, f'{nama} h={h}')
    print('===== (A) dan (C) 97 asal 2022-2023')
    for nama, s in (('TimesFM', 'timesfm'), ('Moirai-2', 'moirai')):
        Q = np.load(os.path.join(T, f'pred_{s}_pra.npy'))
        for h in HORIZON:
            d = G.siapkan(pra, Q, h)
            ta, ts = tgl.to_numpy()[pra], tgl.to_numpy()[pra + h]
            kl, ku = np.flatnonzero(ts < batas), np.flatnonzero(ta >= batas)
            a, b = P.potong(d, kl), P.potong(d, ku)
            b['ok'] = b['ok'] & np.isfinite(b['q']).all(2)
            x, pm = pembanding(a, b, len(ku), 20270200 + h, f'{nama} h={h} 2022-2023')
            m = b['q'][:, :, 1]
            idx = G.boot(len(ku), 20270300 + h)
            print('    campuran dikurangi keras: ' + P.bedakan(x, b, np.where(pm > 0.5, b['c'], m), idx))
            print('    campuran dikurangi rerata: ' + P.bedakan(x, b, pm * b['c'] + (1 - pm) * m, idx))
            kelompok(x, m, b, idx, '2022-23')

    print('===== (B) kelompok, keenam peramal saraf, 142 asal, h = 7/14/28')
    Qp = np.load(os.path.join(T, 'pred_timesfm_pra.npy'))
    Qu = np.load(os.path.join(T, 'pred_timesfm.npy'))
    for h in HORIZON:
        a, b = G.siapkan(pra, Qp, h), G.siapkan(uji, Qu, h)
        pm, _ = dasar(a, len(uji))
        idx = G.boot(len(uji), 20270400 + h)
        print(f'--- h={h}')
        for nama, s in L.SEMUA[:6]:
            q = np.sort(np.load(os.path.join(T, f'pred_{s}.npy'))[:, :, h - 1, :].astype(float), axis=2)
            kelompok(G.median_campuran(pm, b['c'], q), q[:, :, 1], b, idx, nama)
        x, m = G.median_campuran(pm, b['c'], b['q']), b['q'][:, :, 1]
        ok, y, c, sk = b['ok'], b['y'], b['c'], b['skala']
        e = lambda f: np.where(ok, np.abs(y - f), np.nan)
        print('  TimesFM per komoditas lawan naif, MASE saja | campuran | selisih campuran-saja')
        for k_ in KOM:
            r = kom == k_
            print(f'    {k_:24s} {G.ci(np.nanmean(((e(m) - e(c)) / sk)[r], 0), idx)} | {G.ci(np.nanmean(((e(x) - e(c)) / sk)[r], 0), idx)} | '
                  f'{G.ci(np.nanmean(((e(x) - e(m)) / sk)[r], 0), idx)}')

        # (D) skala tanggal survei
        b2 = dict(b)
        b2['skala'] = skala_survei(uji, Y, G.survei)
        print(f'  (D) skala survei saja, nol {int((b2["skala"] <= 0).sum())}; median rasio skala baru/lama lengket '
              f'{np.nanmedian((b2["skala"] / b["skala"])[np.isin(kom, LENGKET)]):.3f} bergerak {np.nanmedian((b2["skala"] / b["skala"])[np.isin(kom, BERGERAK)]):.3f}')
        print('      campuran lawan naif: ' + P.bedakan(x, b2, c, idx))
        print('      campuran lawan TimesFM: ' + P.bedakan(x, b2, m, idx))
        # (D) ambang
        chili = pd.Series(kom).str.contains('cabai', case=False).to_numpy()
        dY = np.diff(Y[chili], axis=1)
        frac = np.r_[np.nan, np.nanmean(np.where(np.isfinite(dY), dY == 0, np.nan), axis=0)]
        for amb in (0.5, 0.7, 0.8):
            sv = ~(frac >= amb)
            b3 = dict(b)
            b3['ok'] = np.isfinite(y) & (b['skala'] > 0) & sv[uji + h][None, :] & A[:, uji + h]
            print(f'      ambang {amb}: tanggal tanpa survei {int((~sv).sum())}; pasangan {int(b3["ok"].sum())}; lawan naif ' + P.bedakan(x, b3, c, idx)
                  + '; lawan TimesFM ' + P.bedakan(x, b3, m, idx))
        # (E) peringatan
        ev = y / c - 1 >= 0.05
        for nm, f in (('TimesFM', m), ('campuran', x)):
            al = f / c - 1 >= 0.05
            print(f'  (E) {nm}: kejadian {ev[ok].mean():.3f}; tangkap {(al & ev)[ok].sum() / ev[ok].sum():.3f}; '
                  f'palsu {(al & ~ev)[ok].sum() / (~ev)[ok].sum():.3f}; presisi {(al & ev)[ok].sum() / max(al[ok].sum(), 1):.3f}; bendera {al[ok].mean():.3f}')

    print('===== (D) laju per tahun')
    th = tgl.dt.year.to_numpy()
    for t in np.unique(th):
        r = th == t
        ss = [np.nanmean(np.where(np.isfinite(np.diff(Y[kom == k_][:, r], axis=1)), np.diff(Y[kom == k_][:, r], axis=1) == 0, np.nan)) for k_ in LENGKET]
        print(f'  {t}: tanggal {int(r.sum())}, tanpa survei {int((~G.survei[r]).sum())} ({(~G.survei[r]).mean() * 100:.1f} %); bagian tak berubah harian ' +
              ' '.join(f'{k[:6]}={v:.3f}' for k, v in zip(LENGKET, ss)))
    ev = np.zeros(len(tgl), bool)
    ev[uji[0]:uji[-1] + 29] = True
    print(f'  rentang evaluasi {(~G.survei[ev]).mean() * 100:.1f} %, di luar {(~G.survei[~ev]).mean() * 100:.1f} %')

    print('===== (F) 5 asal belum dinilai, p beku v1.0.0')
    with open(os.path.join(D, 'campuran_rincian.json'), encoding='utf-8') as f:
        pk = json.load(f)
    Qb = np.load(os.path.join(T, 'pred_timesfm_baru.npy'))
    print('  asal', [str(tgl[i].date()) for i in baru])
    for h in (7, 14):
        k = np.flatnonzero(baru + h < len(tgl))
        b = G.siapkan(baru[k], Qb[:, k], h)
        pm = np.repeat(np.array([pk[str(h)]['latih'][j] for j in kom])[:, None], len(k), 1)
        x, m, c, y, ok, sk = G.median_campuran(pm, b['c'], b['q']), b['q'][:, :, 1], b['c'], b['y'], b['ok'], b['skala']
        e = lambda f: np.abs(y - f)[ok]
        s = sk[ok]
        print(f'  h={h}: asal {len(k)}, pasangan {int(ok.sum())}; MASE naif {np.mean(e(c) / s):.3f} TimesFM {np.mean(e(m) / s):.3f} campuran {np.mean(e(x) / s):.3f}; '
              f'MAPE naif {np.mean(e(c) / y[ok]) * 100:.3f} TimesFM {np.mean(e(m) / y[ok]) * 100:.3f} campuran {np.mean(e(x) / y[ok]) * 100:.3f}')
        for nama, g in (('lengket', LENGKET), ('bergerak', BERGERAK)):
            r = np.isin(kom, g)[:, None] & ok
            print(f'    {nama}: MASE naif {np.mean((np.abs(y - c) / sk)[r]):.3f} TimesFM {np.mean((np.abs(y - m) / sk)[r]):.3f} campuran {np.mean((np.abs(y - x) / sk)[r]):.3f}; '
                  f'MAE Rp {np.mean(np.abs(y - c)[r]):.0f} {np.mean(np.abs(y - m)[r]):.0f} {np.mean(np.abs(y - x)[r]):.0f}')


if __name__ == '__main__':
    main()
