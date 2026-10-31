"""PIHPS — rincian gerbang empat ciri untuk naskah: pembanding beku, kalibrasi, kelompok, per komoditas, Idulfitri.

Melengkapi 19_gerbang_lintas_model.py dengan pembanding yang sama seperti 16_gerbang.py dan 18_periksa_gerbang.py,
kini untuk gerbang empat ciri. Tidak ada keputusan baru; menulis gerbang_rincian.json untuk gambar naskah.

Pakai: venv-gpu python 20_gerbang_rincian.py > gerbang_rincian.log
"""
import importlib.util
import json
import os

import numpy as np
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
s_ = importlib.util.spec_from_file_location('lintas', os.path.join(D, '19_gerbang_lintas_model.py'))
L = importlib.util.module_from_spec(s_)
s_.loader.exec_module(L)
P, G, T, tgl, HORIZON, KOLOM, kom = L.P, L.G, L.T, L.tgl, L.HORIZON, L.KOLOM, L.G.kom


def banding(gm, b, a, idx):
    f1, f2, tz, lis = P.beku(a, b)
    q5 = b['q'][:, :, 1]
    tahan = np.where(np.abs(q5 - b['c']) < 50, b['c'], np.round(q5 / 50) * 50)
    print(f'  ambang beku {tz:.1f}; daftar {lis}')
    print('  kelengketan lawan naif: ' + P.bedakan(f1, b, b['c'], idx))
    print('  daftar lawan naif:      ' + P.bedakan(f2, b, b['c'], idx))
    print('  gerbang lawan kelengketan: ' + P.bedakan(gm, b, f1, idx))
    print('  gerbang lawan daftar:      ' + P.bedakan(gm, b, f2, idx))
    print('  gerbang lawan ditahan Rp 50: ' + P.bedakan(gm, b, tahan, idx), flush=True)


def main():
    pra = pd.read_csv(os.path.join(T, 'asal_pra.csv')).indeks.to_numpy()
    uji = pd.read_csv(os.path.join(T, 'asal.csv')).indeks.to_numpy()
    Qp, Qu = np.load(os.path.join(T, 'pred_timesfm_pra.npy')), np.load(os.path.join(T, 'pred_timesfm.npy'))
    keluar = {'kalibrasi': {}}
    for h in HORIZON:
        a, b = G.siapkan(pra, Qp, h), G.siapkan(uji, Qu, h)
        ok, c, q, y = b['ok'], b['c'], b['q'], b['y']
        p = P.peluang(P.latih(a, KOLOM), b, KOLOM)
        gm = G.median_campuran(p, c, q)
        idx = G.boot(len(uji), 20261900 + h)
        lb, la = b['lab'][ok], a['lab'][a['ok']]
        print(f'\n===== 142 asal, h={h}: diam {lb.mean():.3f}; AUC {G.auc(p[ok], lb):.3f}; Brier {np.mean((p[ok] - lb) ** 2):.4f} '
              f'(dasar {np.mean((la.mean() - lb) ** 2):.4f})')
        banding(gm, b, a, idx)
        s = pd.DataFrame({'p': p[ok], 'l': lb})
        s['d'] = pd.qcut(s.p, 10, labels=False, duplicates='drop')
        k = s.groupby('d').mean()
        keluar['kalibrasi'][str(h)] = {'p': k.p.round(4).tolist(), 'f': k.l.round(4).tolist()}
        print('  kalibrasi: ' + ' '.join(f'{r.p:.2f}/{r.l:.2f}' for r in k.itertuples()))
        print(f'  tepat di harga asal: semua {np.mean((gm == c)[ok]) * 100:.1f} %')
        d = np.where(ok, np.abs(y - gm) - np.abs(y - c), np.nan)
        dm = np.where(ok, np.abs(y - q[:, :, 1]) - np.abs(y - c), np.nan)
        dt = np.where(ok, np.abs(y - gm) - np.abs(y - q[:, :, 1]), np.nan)
        for nama, kel in (('bergerak', P.BERGERAK), ('lengket', G.LENGKET)):
            r = np.isin(kom, kel)
            print(f'  {nama}: tepat di harga asal {np.mean((gm == c)[r][ok[r]]) * 100:.1f} %; lawan naif MAPE {G.ci(np.nanmean((d / y * 100)[r], 0), idx)} '
                  f'MASE {G.ci(np.nanmean((d / b["skala"])[r], 0), idx)}; TimesFM lawan naif MAPE {G.ci(np.nanmean((dm / y * 100)[r], 0), idx)}; '
                  f'lawan TimesFM MAPE {G.ci(np.nanmean((dt / y * 100)[r], 0), idx)}')
        print('  per komoditas, lawan naif MAPE / MASE')
        for kk in np.unique(kom):
            r = kom == kk
            print(f'    {kk:24s} {G.ci(np.nanmean((d / y * 100)[r], 0), idx)} / {G.ci(np.nanmean((d / b["skala"])[r], 0), idx)}')
        ts = tgl.to_numpy()[uji + h]
        print('  Idulfitri (sasaran -30..+7 hari), lawan naif MAPE gerbang | TimesFM:')
        for hr in pd.to_datetime(['2024-04-10', '2025-03-31', '2026-03-21']):
            m = (ts >= np.datetime64(hr - pd.Timedelta(days=30))) & (ts <= np.datetime64(hr + pd.Timedelta(days=7)))
            print(f'    {hr.date()} ({m.sum()} asal): {np.nanmean((d / y * 100)[:, m]):+.3f} | {np.nanmean((dm / y * 100)[:, m]):+.3f}')
    batas = np.datetime64('2022-01-01')
    d0 = None
    for nama, s in (('timesfm', 'timesfm'), ('moirai', 'moirai')):
        Q = np.load(os.path.join(T, f'pred_{s}_pra.npy'))
        for h in HORIZON:
            d0 = G.siapkan(pra, Q, h)
            ta, ts = tgl.to_numpy()[pra], tgl.to_numpy()[pra + h]
            kl, ku = np.flatnonzero(ts < batas), np.flatnonzero(ta >= batas)
            a, b = P.potong(d0, kl), P.potong(d0, ku)
            b['ok'] = b['ok'] & np.isfinite(b['q']).all(2)
            p = P.peluang(P.latih(a, KOLOM), b, KOLOM)
            gm = G.median_campuran(p, b['c'], b['q'])
            print(f'\n===== 97 asal {nama}, h={h}: AUC {G.auc(p[b["ok"]], b["lab"][b["ok"]]):.3f}')
            banding(gm, b, a, G.boot(len(ku), 20262000 + h))
    with open(os.path.join(D, 'gerbang_rincian.json'), 'w', encoding='utf-8') as f:
        json.dump(keluar, f)


if __name__ == '__main__':
    main()
