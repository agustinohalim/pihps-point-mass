"""PIHPS — campuran massa titik dengan peluang dasar komoditas: metode utama naskah (keputusan pengguna 31 Okt 2026, opsi A).

Peluang p = frekuensi harga tidak berubah neto pada horizon h, per komoditas, dari data latih saja. Prakiraan = median
campuran p·titik di harga terakhir + (1-p)·distribusi kuantil model (G.median_campuran, tidak berubah). Rancangan di
Hasil_Uji_PIHPS.md ("Rancangan opsi A"). Menulis campuran_rincian.json untuk gambar.

Pakai: venv-gpu python 24_campuran_dasar.py > campuran.log
"""
import importlib.util
import json
import os

import numpy as np
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
s_ = importlib.util.spec_from_file_location('tg', os.path.join(D, '22_gerbang_telaah.py'))
L = importlib.util.module_from_spec(s_)
s_.loader.exec_module(L)
P, G, T, tgl, HORIZON, KOLOM, kom, rmae, smape, boot_blok = L.P, L.G, L.T, L.tgl, L.HORIZON, L.KOLOM, L.kom, L.rmae, L.smape, L.boot_blok
wil = pd.read_csv(os.path.join(T, 'deret.csv')).wilayah.astype(str)
NAS = wil.str.contains('Semua', case=False).to_numpy()
BERGERAK, LENGKET = P.BERGERAK, G.LENGKET


def dasar(a, n):
    d = {k: float(a['lab'][kom == k][a['ok'][kom == k]].mean()) for k in np.unique(kom)}
    return np.repeat(np.array([d[k] for k in kom])[:, None], n, 1), d


def rupiah(x, b, s):
    r = np.isin(kom, s)[:, None] & b['ok']
    return float(np.mean(np.abs(b['y'] - x)[r]))


def utama():
    pra = pd.read_csv(os.path.join(T, 'asal_pra.csv')).indeks.to_numpy()
    uji = pd.read_csv(os.path.join(T, 'asal.csv')).indeks.to_numpy()
    Qp, Qu = np.load(os.path.join(T, 'pred_timesfm_pra.npy')), np.load(os.path.join(T, 'pred_timesfm.npy'))
    tahun = pd.DatetimeIndex(tgl.to_numpy()[uji]).year.to_numpy()
    keluar = {}
    for h in HORIZON:
        a, b = G.siapkan(pra, Qp, h), G.siapkan(uji, Qu, h)
        ok, c, q, y = b['ok'], b['c'], b['q'], b['y']
        pm, pk = dasar(a, len(uji))
        amat = {k: float(b['lab'][kom == k][ok[kom == k]].mean()) for k in np.unique(kom)}
        keluar[str(h)] = {'latih': pk, 'uji': amat}
        idx = G.boot(len(uji), 20262700 + h)
        f1, f2, tz, lis = P.beku(a, b)
        dipilih = np.isin(kom, lis)
        print(f'\n===== h={h}, 142 asal')
        print('  p latih -> amat: ' + ' '.join(f'{k[:9]}={pk[k]:.3f}>{amat[k]:.3f}' for k in pk))
        print(f'  daftar {lis}; ambang {tz:.1f}')
        for nama, s in L.SEMUA:
            Q = np.sort(np.load(os.path.join(T, f'pred_{s}.npy'))[:, :, h - 1, :].astype(float), axis=2)
            m = Q[:, :, 1]
            x = G.median_campuran(pm, c, Q)
            print(f'  {nama}')
            print('    saja lawan naif:   ' + P.bedakan(m, b, c, idx))
            print('    campuran lawan naif:' + P.bedakan(x, b, c, idx))
            print('    campuran lawan saja:' + P.bedakan(x, b, m, idx))
            print(f'    rMAE {rmae(m, b, idx)} -> {rmae(x, b, idx)}; sMAPE {smape(m, b):.3f} -> {smape(x, b):.3f}', flush=True)
            print('    campuran lawan daftar pinjaman: ' + P.bedakan(x, b, np.where(dipilih[:, None], m, c), idx))
        x = G.median_campuran(pm, c, q)
        m = q[:, :, 1]
        pg = P.peluang(P.latih(a, KOLOM), b, KOLOM)
        gm = G.median_campuran(pg, c, q)
        pkeras, prata = np.where(pm > 0.5, c, m), pm * c + (1 - pm) * m
        tahan = np.where(np.abs(m - c) < 50, c, np.round(m / 50) * 50)
        print('  TimesFM, campuran dikurangi pembanding:')
        for nama, f in (('naif', c), ('TimesFM', m), ('LightGBM', gm), ('kelengketan', f1), ('daftar', f2), ('tahan Rp 50', tahan),
                        ('keras', pkeras), ('rerata', prata)):
            print(f'    {nama}: ' + P.bedakan(x, b, f, idx))
        print(f'  tepat di harga asal: semua {np.mean((x == c)[ok]) * 100:.1f} %')
        d, dm = (np.where(ok, np.abs(y - f) - np.abs(y - c), np.nan) for f in (x, m))
        for nama, kel in (('bergerak', BERGERAK), ('lengket', LENGKET)):
            r = np.isin(kom, kel)
            print(f'  {nama}: tepat {np.mean((x == c)[r][ok[r]]) * 100:.1f} %; campuran MAPE {G.ci(np.nanmean((d / y * 100)[r], 0), idx)} '
                  f'MASE {G.ci(np.nanmean((d / b["skala"])[r], 0), idx)}; TimesFM MAPE {G.ci(np.nanmean((dm / y * 100)[r], 0), idx)} '
                  f'MASE {G.ci(np.nanmean((dm / b["skala"])[r], 0), idx)}; MAE rupiah naif {rupiah(c, b, kel):.0f}, TimesFM {rupiah(m, b, kel):.0f}, campuran {rupiah(x, b, kel):.0f}')
        print('  per komoditas lawan naif MAPE: TimesFM | campuran; MAE rupiah TimesFM -> campuran')
        for kk in np.unique(kom):
            r = kom == kk
            print(f'    {kk:24s} {G.ci(np.nanmean((dm / y * 100)[r], 0), idx)} | {G.ci(np.nanmean((d / y * 100)[r], 0), idx)}; {rupiah(m, b, [kk]):.1f} -> {rupiah(x, b, [kk]):.1f}')
        print('  per tahun asal, campuran lawan naif')
        for th in (2024, 2025, 2026):
            k = tahun == th
            bb = P.potong({v: b[v] for v in ('ok', 'y', 'skala')}, k)
            print(f'    {th}: ' + P.bedakan(x[:, k], bb, c[:, k], G.boot(int(k.sum()), 20262800 + h)))
        bn = dict(b)
        bn['ok'] = ok & ~NAS[:, None]
        print('    tanpa nasional: ' + P.bedakan(x, bn, c, idx))
        print('    blok 16: ' + P.bedakan(x, b, c, boot_blok(len(uji), 20262900 + h, 16)))
        ts = tgl.to_numpy()[uji + h]
        for hr in pd.to_datetime(['2024-04-10', '2025-03-31', '2026-03-21']):
            w = (ts >= np.datetime64(hr - pd.Timedelta(days=30))) & (ts <= np.datetime64(hr + pd.Timedelta(days=7)))
            print(f'    Idulfitri {hr.date()} ({w.sum()} asal) MAPE campuran | TimesFM: {np.nanmean((d / y * 100)[:, w]):+.3f} | {np.nanmean((dm / y * 100)[:, w]):+.3f}')
    with open(os.path.join(D, 'campuran_rincian.json'), 'w', encoding='utf-8') as f:
        json.dump(keluar, f)


def awal():
    pra = pd.read_csv(os.path.join(T, 'asal_pra.csv')).indeks.to_numpy()
    batas = np.datetime64('2022-01-01')
    print('\n===== 97 asal 2022-2023, p dari sasaran < 2022')
    for nama, s in (('TimesFM', 'timesfm'), ('Moirai-2', 'moirai')):
        Q = np.load(os.path.join(T, f'pred_{s}_pra.npy'))
        for h in HORIZON:
            d = G.siapkan(pra, Q, h)
            ta, ts = tgl.to_numpy()[pra], tgl.to_numpy()[pra + h]
            kl, ku = np.flatnonzero(ts < batas), np.flatnonzero(ta >= batas)
            a, b = P.potong(d, kl), P.potong(d, ku)
            b['ok'] = b['ok'] & np.isfinite(b['q']).all(2)
            c, q, m = b['c'], b['q'], b['q'][:, :, 1]
            pm, _ = dasar(a, len(ku))
            x = G.median_campuran(pm, c, q)
            f1, f2, tz, _ = P.beku(a, b)
            tahan = np.where(np.abs(m - c) < 50, c, np.round(m / 50) * 50)
            gm = G.median_campuran(P.peluang(P.latih(a, KOLOM), b, KOLOM), c, q)
            idx = G.boot(len(ku), 20263000 + h)
            print(f'{nama} h={h}, ambang {tz:.1f}; rMAE {rmae(m, b, idx)} -> {rmae(x, b, idx)}')
            for n, f in (('naif', c), ('model', m), ('LightGBM', gm), ('kelengketan', f1), ('daftar', f2), ('tahan Rp 50', tahan)):
                print(f'    campuran dikurangi {n}: ' + P.bedakan(x, b, f, idx), flush=True)


if __name__ == '__main__':
    utama()
    awal()
