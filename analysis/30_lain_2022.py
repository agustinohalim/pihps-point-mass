"""PIHPS — 12 komoditas tak tersentuh pada asal 2022–2023, prosedur beku (telaah 02f, kritis periode tunggal).

Rancangan ditulis sebelum dijalankan: Hasil_Uji_PIHPS.md, "Rancangan skrip 30". Sama dengan uji 97 asal pada sembilan
komoditas (skrip 24 `awal`, skrip 27 bagian A): p dan pembanding dari sasaran < 2022, dinilai pada asal >= 2022 dari
asal_pra. Hanya TimesFM. Ditambah korelasi perubahan log harian cabai tak tersentuh lawan cabai pengembangan.

Pakai: PIHPS_DATA=C:/Data/PIHPS_lain venv-gpu python 30_lain_2022.py > lain_2022.log
"""
import importlib.util
import os

import numpy as np
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
assert 'PIHPS_lain' in os.environ.get('PIHPS_DATA', ''), 'jalankan dengan PIHPS_DATA=C:/Data/PIHPS_lain'
s_ = importlib.util.spec_from_file_location('ta', os.path.join(D, '27_telaah_akhir.py'))
K = importlib.util.module_from_spec(s_)
s_.loader.exec_module(K)
P, G, T, kom, dasar, HORIZON, tgl = K.P, K.G, K.T, K.kom, K.dasar, K.HORIZON, K.tgl
KOM = np.unique(kom)
KELOMPOK = {'beras': [k for k in KOM if k.startswith('beras')],
            'minyak dan gula': [k for k in KOM if k.startswith(('minyak', 'gula'))],
            'cabai': [k for k in KOM if k.startswith('cabai')],
            'daging sapi': [k for k in KOM if k.startswith('daging_sapi')]}
UTAMA = 'C:/Data/PIHPS/turunan'


def rp(x, b, r):
    # pasangan sama untuk naif, TimesFM, campuran: harga asal ada (sebagian asal 2022 tanpa harga asal)
    k = r & b['ok'] & np.isfinite(b['c'])
    return float(np.mean(np.abs(b['y'] - x)[k]))


def korelasi():
    print('\n===== korelasi perubahan log harian per provinsi, cabai tak tersentuh lawan pengembangan (median antarprovinsi)')
    da, db = pd.read_csv(os.path.join(UTAMA, 'deret.csv')), pd.read_csv(os.path.join(T, 'deret.csv'))
    Ya, Yb = np.load(os.path.join(UTAMA, 'Y.npy')).astype(float), np.load(os.path.join(T, 'Y.npy')).astype(float)
    assert Ya.shape[1] == Yb.shape[1]
    for kb in ('cabai_merah_besar', 'cabai_rawit_hijau'):
        for ka in ('cabai_merah_keriting', 'cabai_rawit_merah', 'bawang_merah'):
            r = []
            for w in db.wilayah[db.komoditas == kb]:
                ia, ib = np.flatnonzero((da.komoditas == ka) & (da.wilayah == w)), np.flatnonzero((db.komoditas == kb) & (db.wilayah == w))
                if len(ia) != 1 or len(ib) != 1:
                    continue
                xa, xb = np.diff(np.log(Ya[ia[0]])), np.diff(np.log(Yb[ib[0]]))
                k = np.isfinite(xa) & np.isfinite(xb)
                if k.sum() > 100 and xa[k].std() > 0 and xb[k].std() > 0:
                    r.append(np.corrcoef(xa[k], xb[k])[0, 1])
            print(f'  {kb} ~ {ka}: median {np.median(r):.3f} (kuartil {np.percentile(r, 25):.3f}–{np.percentile(r, 75):.3f}), {len(r)} provinsi')


def main():
    pra = pd.read_csv(os.path.join(T, 'asal_pra.csv')).indeks.to_numpy()
    batas = np.datetime64('2022-01-01')
    Q = np.load(os.path.join(T, 'pred_timesfm_pra.npy'))
    print(f'deret {len(kom)}; komoditas {list(KOM)}')
    for h in HORIZON:
        d = G.siapkan(pra, Q, h)
        ta, ts = tgl.to_numpy()[pra], tgl.to_numpy()[pra + h]
        kl, ku = np.flatnonzero(ts < batas), np.flatnonzero(ta >= batas)
        a, b = P.potong(d, kl), P.potong(d, ku)
        b['ok'] = b['ok'] & np.isfinite(b['q']).all(2)
        c, q, ok = b['c'], b['q'], b['ok']
        m = q[:, :, 1]
        pm, pk = dasar(a, len(ku))
        x = G.median_campuran(pm, c, q)
        f1, f2, tz, lis = P.beku(a, b)
        kk, ww, pg = K.setel(a)
        z, s, _ = K.pakai(b, kk, ww, pg)
        idx = G.boot(len(ku), 20263100 + h)
        print(f'\n===== h={h}, latih {len(kl)} asal, uji {len(ku)} asal 2022-2023, pasangan dinilai {int(ok.sum())}')
        print('  p latih: ' + ' '.join(f'{k}={pk[k]:.3f}' for k in KOM))
        print(f'  daftar pakai model {lis}; ambang kelengketan {tz:.1f}')
        for t, f in (('TimesFM', m), ('campuran', x), ('daftar', f2), ('kelengketan', f1), ('susut', s), ('zona mati', z)):
            print(f'    {t} dikurangi naif: ' + P.bedakan(f, b, c, idx))
        for t, f in (('TimesFM', m), ('susut', s), ('zona mati', z), ('daftar', f2)):
            print(f'    campuran dikurangi {t}: ' + P.bedakan(x, b, f, idx))
        print('  kelompok: TimesFM-naif MASE | campuran-naif MASE; campuran-TimesFM MASE, MAPE; MAE Rp naif / TimesFM / campuran')
        for nk, ks in list(KELOMPOK.items()) + [(k, [k]) for k in KOM]:
            r = np.isin(kom, ks)[:, None]
            bb = dict(b)
            bb['ok'] = ok & r
            print(f'    {nk:24s} {P.bedakan(m, bb, c, idx).split(" MAPE")[0]} | {P.bedakan(x, bb, c, idx).split(" MAPE")[0]}; '
                  f'{P.bedakan(x, bb, m, idx)}; Rp {rp(c, b, r):.0f} / {rp(m, b, r):.0f} / {rp(x, b, r):.0f}', flush=True)
    korelasi()


if __name__ == '__main__':
    main()
