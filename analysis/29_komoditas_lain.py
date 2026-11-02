"""PIHPS — uji komoditas tak tersentuh: prosedur naskah dibekukan, dinilai pada 12 komoditas PIHPS yang tidak dipakai naskah.

Rancangan ditulis sebelum data diunduh: Hasil_Uji_PIHPS.md, "Rancangan skrip 29". Tidak ada pilihan baru: p dari
`dasar` (24_campuran_dasar), median campuran `G.median_campuran`, pembanding disetel di data latih seperti skrip 27,
daftar dan aturan kelengketan dari `P.beku`, bootstrap dan benih sama dengan skrip 24/27.

Persiapan (semua dengan PIHPS_DATA=C:/Data/PIHPS_lain): 01_unduh.py (PIHPS_KOMODITAS=lain), 02_siapkan.py,
29a_siapkan_lain.py, lalu 03_model_fondasi.py untuk chronos2, chronosbolt, timesfm, moirai, dan timesfm dengan
PIHPS_ASAL=pra (dasar p hanya memakai label; prakiraan pra dipakai untuk menyetel susut, zona mati, dan daftar).

Pakai: PIHPS_DATA=C:/Data/PIHPS_lain venv-gpu python 29_komoditas_lain.py > komoditas_lain.log
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
MODEL = [('Chronos-2', 'chronos2'), ('Chronos-Bolt', 'chronosbolt'), ('TimesFM 2.5', 'timesfm'), ('Moirai-2', 'moirai')]
KELOMPOK = {'beras': [k for k in KOM if k.startswith('beras')],
            'minyak dan gula': [k for k in KOM if k.startswith(('minyak', 'gula'))],
            'cabai': [k for k in KOM if k.startswith('cabai')],
            'daging sapi (kategori baru)': [k for k in KOM if k.startswith('daging_sapi')]}


def rp(x, b, r):
    return float(np.mean(np.abs(b['y'] - x)[r & b['ok']]))


def main():
    pra = pd.read_csv(os.path.join(T, 'asal_pra.csv')).indeks.to_numpy()
    uji = pd.read_csv(os.path.join(T, 'asal.csv')).indeks.to_numpy()
    tahun = pd.DatetimeIndex(tgl.to_numpy()[uji]).year.to_numpy()
    print(f'deret {len(kom)}; komoditas {list(KOM)}')
    Qp, Qu = np.load(os.path.join(T, 'pred_timesfm_pra.npy')), np.load(os.path.join(T, 'pred_timesfm.npy'))
    for h in HORIZON:
        a, b = G.siapkan(pra, Qp, h), G.siapkan(uji, Qu, h)
        c, ok = b['c'], b['ok']
        pm, pk = dasar(a, len(uji))
        amat = {k: float(b['lab'][kom == k][ok[kom == k]].mean()) for k in KOM}
        idx = G.boot(len(uji), 20262700 + h)
        print(f'\n===== h={h}, 142 asal, pasangan dinilai {int(ok.sum())}')
        print('  p latih -> amat: ' + ' '.join(f'{k}={pk[k]:.3f}>{amat[k]:.3f}' for k in KOM))
        print('  H1/H2/H3: per model')
        for nama, s in MODEL:
            q = np.sort(np.load(os.path.join(T, f'pred_{s}.npy'))[:, :, h - 1, :].astype(float), axis=2)
            m, x = q[:, :, 1], G.median_campuran(pm, c, q)
            print(f'  {nama}')
            print('    saja lawan naif:     ' + P.bedakan(m, b, c, idx))
            print('    campuran lawan naif: ' + P.bedakan(x, b, c, idx))
            print('    campuran lawan saja: ' + P.bedakan(x, b, m, idx))
            print(f'    rMAE {K.L.rmae(m, b, idx)} -> {K.L.rmae(x, b, idx)}', flush=True)
        # H4 dan rincian: TimesFM
        q, m = b['q'], b['q'][:, :, 1]
        x = G.median_campuran(pm, c, q)
        kk, ww, pg = K.setel(a)
        z, s_, pgm = K.pakai(b, kk, ww, pg)
        gl = G.median_campuran(pgm, c, q)
        f1, f2, tz, lis = P.beku(a, b)
        print(f'  H4 TimesFM: k {[kk[k] for k in KOM]}; w {[ww[k] for k in KOM]}; p global {pg:.3f}; daftar pakai model {lis}; ambang {tz:.1f}')
        for t, f in (('naif', c), ('TimesFM', m), ('susut', s_), ('zona mati', z), ('p global', gl), ('daftar', f2), ('kelengketan', f1)):
            print(f'    campuran dikurangi {t}: ' + P.bedakan(x, b, f, idx))
        for t, f in (('daftar', f2), ('kelengketan', f1), ('susut', s_), ('zona mati', z)):
            print(f'    {t} dikurangi naif: ' + P.bedakan(f, b, c, idx))
        print(f'  tepat di harga asal: {np.mean((x == c)[ok]) * 100:.1f} %')
        print('  kelompok: lawan naif MASE TimesFM | campuran; campuran-TimesFM MASE, MAPE; MAE Rp naif / TimesFM / campuran')
        for nk, ks in list(KELOMPOK.items()) + [(k, [k]) for k in KOM]:
            r = np.isin(kom, ks)[:, None]
            bb = dict(b)
            bb['ok'] = ok & r
            print(f'    {nk:28s} {P.bedakan(m, bb, c, idx).split(" MAPE")[0]} | {P.bedakan(x, bb, c, idx).split(" MAPE")[0]}; '
                  f'{P.bedakan(x, bb, m, idx)}; Rp {rp(c, b, r):.0f} / {rp(m, b, r):.0f} / {rp(x, b, r):.0f}', flush=True)
        print('  per tahun asal, campuran lawan TimesFM | lawan naif')
        for th in (2024, 2025, 2026):
            k = tahun == th
            bb = P.potong({v: b[v] for v in ('ok', 'y', 'skala')}, k)
            ii = G.boot(int(k.sum()), 20262800 + h)
            print(f'    {th}: {P.bedakan(x[:, k], bb, m[:, k], ii)} | {P.bedakan(x[:, k], bb, c[:, k], ii)}')


if __name__ == '__main__':
    main()
