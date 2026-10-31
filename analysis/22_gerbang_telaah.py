"""PIHPS — uji gerbang yang diminta telaah 31c (rancangan di Hasil_Uji_PIHPS.md, ditulis sebelum dijalankan).

(1) rMAE dan sMAPE, sendiri lawan bergerbang, semua peramal termasuk ETS dan ARIMA.
(2) Ambang dan daftar yang diset dari TimesFM dipinjam ke semua peramal; peluang dasar komoditas tanpa pengklasifikasi.
(3) AUC dalam komoditas dan kalibrasi per komoditas.
(4) Keras, rerata, median pada kedua periode; pilihan dibuat pada 2022-2023 sebelum melihat 2024-2026.
(5) Per tahun sasaran dan tanpa deret nasional, bergerbang.
(6) Bootstrap blok 16.

Pakai: venv-gpu python 22_gerbang_telaah.py > gerbang_telaah.log
"""
import importlib.util
import os

import numpy as np
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
s_ = importlib.util.spec_from_file_location('lintas', os.path.join(D, '19_gerbang_lintas_model.py'))
L = importlib.util.module_from_spec(s_)
s_.loader.exec_module(L)
P, G, T, tgl, HORIZON, KOLOM, kom = L.P, L.G, L.T, L.tgl, L.HORIZON, L.KOLOM, L.G.kom
deret = pd.read_csv(os.path.join(T, 'deret.csv'))
NAS = deret.iloc[:, 0].astype(str).str.lower().str.contains('nasional').to_numpy() | deret.apply(
    lambda r: r.astype(str).str.lower().str.contains('nasional').any(), axis=1).to_numpy()
SEMUA = L.MODEL + [('ETS', 'ets'), ('ARIMA', 'arima')]


def boot_blok(ban, seed, blok):
    rng = np.random.default_rng(seed)
    aw = rng.integers(0, ban - blok + 1, (10000, int(np.ceil(ban / blok))))
    return (aw[:, :, None] + np.arange(blok)).reshape(10000, -1)[:, :ban]


def rmae(x, b, idx):
    ok, y = b['ok'], b['y']
    e, n = np.where(ok, np.abs(y - x), np.nan), np.where(ok, np.abs(y - b['c']), np.nan)
    f = lambda v, w: np.exp(np.nanmean(np.log(np.nanmean(v, 1) / np.nanmean(w, 1))))
    s = [f(e[:, i], n[:, i]) for i in idx[:2000]]
    return f'{f(e, n):.3f} [{np.percentile(s, 2.5):.3f}; {np.percentile(s, 97.5):.3f}]'


def smape(x, b):
    ok, y = b['ok'], b['y']
    return float(np.nanmean(np.where(ok, 200 * np.abs(y - x) / (np.abs(y) + np.abs(x)), np.nan)))


def main():
    pra = pd.read_csv(os.path.join(T, 'asal_pra.csv')).indeks.to_numpy()
    uji = pd.read_csv(os.path.join(T, 'asal.csv')).indeks.to_numpy()
    Qp, Qu = np.load(os.path.join(T, 'pred_timesfm_pra.npy')), np.load(os.path.join(T, 'pred_timesfm.npy'))
    print('deret nasional', int(NAS.sum()))
    tahun = pd.DatetimeIndex(tgl.to_numpy()[uji]).year.to_numpy()
    for h in HORIZON:
        a, b = G.siapkan(pra, Qp, h), G.siapkan(uji, Qu, h)
        ok, c = b['ok'], b['c']
        p = P.peluang(P.latih(a, KOLOM), b, KOLOM)
        idx = G.boot(len(uji), 20262100 + h)
        _, _, tz, lis = P.beku(a, b)
        dipilih = np.isin(kom, lis)
        dasar = np.array([a['lab'][kom == k][a['ok'][kom == k]].mean() for k in kom])
        pd_ = np.repeat(dasar[:, None], len(uji), 1)
        print(f'\n===== h={h}: ambang TimesFM {tz:.1f}; daftar TimesFM {lis}')
        print('(1)(2) peramal: rMAE saja | bergerbang; sMAPE saja | bergerbang; lawan naif MASE/MAPE untuk gerbang, pinjam ambang, pinjam daftar, peluang dasar')
        for nama, s in SEMUA:
            q = np.sort(np.load(os.path.join(T, f'pred_{s}.npy'))[:, :, h - 1, :].astype(float), axis=2)
            m = q[:, :, 1]
            gm = G.median_campuran(p, c, q)
            pa = np.where(b['X'][:, :, 0] < tz, m, c)
            pl = np.where(dipilih[:, None], m, c)
            bd = G.median_campuran(pd_, c, q)
            print(f'  {nama}: rMAE {rmae(m, b, idx)} | {rmae(gm, b, idx)}; sMAPE {smape(m, b):.3f} | {smape(gm, b):.3f}')
            print('    gerbang:       ' + P.bedakan(gm, b, c, idx))
            print('    pinjam ambang: ' + P.bedakan(pa, b, c, idx))
            print('    pinjam daftar: ' + P.bedakan(pl, b, c, idx))
            print('    peluang dasar: ' + P.bedakan(bd, b, c, idx))
            print('    gerbang lawan pinjam daftar: ' + P.bedakan(gm, b, pl, idx))
            print('    gerbang lawan pinjam ambang: ' + P.bedakan(gm, b, pa, idx))
            print('    gerbang lawan peluang dasar: ' + P.bedakan(gm, b, bd, idx), flush=True)
        print('(3) dalam komoditas: AUC gerbang | AUC kelengketan saja | selisih kalibrasi rerata (ECE 10 desil)')
        lb, zz = b['lab'], b['X'][:, :, 0]
        for kk in np.unique(kom):
            r = (kom == kk)[:, None] & ok
            s = pd.DataFrame({'p': p[r], 'l': lb[r]})
            s['d'] = pd.qcut(s.p, 10, labels=False, duplicates='drop')
            g = s.groupby('d').agg(p=('p', 'mean'), l=('l', 'mean'), n=('l', 'size'))
            ece = float((np.abs(g.p - g.l) * g.n).sum() / g.n.sum())
            print(f'    {kk:24s} diam {lb[r].mean():.3f} AUC {G.auc(p[r], lb[r]):.3f} | {G.auc(zz[r], lb[r]):.3f} | ECE {ece:.3f}')
        q = b['q']
        m = q[:, :, 1]
        print('(4) cara gabung, TimesFM, 142 asal')
        for nama, x in (('keras', np.where(p > 0.5, c, m)), ('rerata', p * c + (1 - p) * m), ('median', G.median_campuran(p, c, q))):
            print(f'    {nama}: lawan naif {P.bedakan(x, b, c, idx)}')
        gm = G.median_campuran(p, c, q)
        print('(5) per tahun asal dan tanpa nasional, gerbang TimesFM lawan naif')
        for th in (2024, 2025, 2026):
            k = tahun == th
            bb = P.potong({x: b[x] for x in ('ok', 'y', 'skala')}, k)
            print(f'    {th} ({k.sum()} asal): ' + P.bedakan(gm[:, k], bb, c[:, k], G.boot(int(k.sum()), 20262200 + h)))
        bn = dict(b)
        bn['ok'] = ok & ~NAS[:, None]
        print('    tanpa nasional: ' + P.bedakan(gm, bn, c, idx))
        i16 = boot_blok(len(uji), 20262300 + h, 16)
        f1, f2, _, _ = P.beku(a, b)
        print('(6) blok 16: lawan naif ' + P.bedakan(gm, b, c, i16) + ' | lawan daftar ' + P.bedakan(gm, b, f2, i16))
    print('\n===== (4) cara gabung, 97 asal 2022-2023, TimesFM')
    batas = np.datetime64('2022-01-01')
    for h in HORIZON:
        d = G.siapkan(pra, Qp, h)
        ta, ts = tgl.to_numpy()[pra], tgl.to_numpy()[pra + h]
        kl, ku = np.flatnonzero(ts < batas), np.flatnonzero(ta >= batas)
        a, b = P.potong(d, kl), P.potong(d, ku)
        b['ok'] = b['ok'] & np.isfinite(b['q']).all(2)
        p = P.peluang(P.latih(a, KOLOM), b, KOLOM)
        idx = G.boot(len(ku), 20262400 + h)
        c, q, m = b['c'], b['q'], b['q'][:, :, 1]
        for nama, x in (('keras', np.where(p > 0.5, c, m)), ('rerata', p * c + (1 - p) * m), ('median', G.median_campuran(p, c, q))):
            print(f'  h={h} {nama}: lawan naif {P.bedakan(x, b, c, idx)}')


if __name__ == '__main__':
    main()
