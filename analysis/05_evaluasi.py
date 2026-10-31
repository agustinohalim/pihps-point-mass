"""PIHPS — metrik untuk semua model (rencana: ../Artikel_PIHPS_Rencana.md, bagian A dan D).

Metrik pada horizon 1, 7, 14, 28 hari kerja, per pasangan (deret, asal) yang sasarannya teramati:
- MASE (utama): |y − ŷ| dibagi rerata |y(t) − y(t−1)| dalam konteks 512 hari kerja deret itu pada asal
  itu (galat naif satu langkah di dalam sampel). Pasangan dengan skala 0 (harga tidak pernah berubah
  dalam konteks) dikeluarkan dan dihitung.
- sMAPE (%): 200 · |y − ŷ| / (|y| + |ŷ|).
- Cakupan interval 80 % (kuantil 0,1–0,9) untuk model yang punya kuantil.
Semua model dinilai pada himpunan pasangan yang sama (horizon yang sama, sasaran teramati, prakiraan ada
untuk semua model).

Ketidakpastian: selisih MASE berpasangan terhadap naif dan terhadap pembanding terlatih terbaik, CI95
bootstrap blok atas asal (blok 4 asal berurutan ≈ 1 bulan, 1.000 ulangan).
Jendela Idulfitri: tanggal sasaran H−30 s.d. H+7 hari kalender terhadap 1 Syawal penetapan pemerintah
2024–2026 (Hasil_Uji_PIHPS.md).

Pakai: python 05_evaluasi.py > evaluasi.log
"""
import os

import numpy as np
import pandas as pd

D = os.environ.get('PIHPS_DATA', 'C:/Data/PIHPS')
T = os.path.join(D, 'turunan')
KONTEKS, HORIZON, BOOT, BLOK = 512, [1, 7, 14, 28], 1000, 4
FONDASI = ['chronos2', 'chronosbolt', 'timesfm', 'moirai']
TERLATIH = ['naif', 'naifmusim', 'ets', 'arima', 'lgbm']
SYAWAL = pd.to_datetime(['2024-04-10', '2025-03-31', '2026-03-21'])


def main():
    Y = np.load(os.path.join(T, 'Y.npy'))
    asal = pd.read_csv(os.path.join(T, 'asal.csv')).indeks.to_numpy()
    tgl = pd.to_datetime(pd.read_csv(os.path.join(T, 'tanggal.csv')).tanggal).to_numpy()
    kom = pd.read_csv(os.path.join(T, 'deret.csv')).komoditas.to_numpy()
    model = [m for m in FONDASI + TERLATIH if os.path.exists(os.path.join(T, f'pred_{m}.npy'))]
    P = {m: np.load(os.path.join(T, f'pred_{m}.npy')) for m in model}
    print(f'model: {", ".join(model)}')
    n = Y.shape[0]
    skala = np.array([[np.nanmean(np.abs(np.diff(Y[i, o - KONTEKS + 1:o + 1]))) for o in asal] for i in range(n)])
    nol = skala == 0
    print(f'pasangan (deret, asal) berskala 0 dikeluarkan: {int(nol.sum())} dari {nol.size}')

    baris = []
    for h in HORIZON:
        y = Y[:, asal + h]
        ok = np.isfinite(y) & ~nol & np.isfinite(skala)
        for m in model:
            ok &= np.isfinite(P[m][:, :, h - 1, 1])
        t_sas = tgl[asal + h]
        idul = np.zeros(len(asal), bool)
        for s in SYAWAL:
            dd = (t_sas - s.to_datetime64()).astype('timedelta64[D]').astype(int)
            idul |= (dd >= -30) & (dd <= 7)
        print(f'\nh = {h}: pasangan dinilai {int(ok.sum())}; asal dengan sasaran di jendela Idulfitri {int(idul.sum())}')
        mase = {m: np.abs(y - P[m][:, :, h - 1, 1]) / skala for m in model}
        smape = {m: 200 * np.abs(y - P[m][:, :, h - 1, 1]) / (np.abs(y) + np.abs(P[m][:, :, h - 1, 1])) for m in model}
        # MASE per asal (rerata atas deret), dasar bootstrap blok
        per_asal = {m: np.array([np.nanmean(np.where(ok[:, k], mase[m][:, k], np.nan)) for k in range(len(asal))])
                    for m in model}
        rng = np.random.default_rng(20261028 + h)
        nb = int(np.ceil(len(asal) / BLOK))
        idx = [np.concatenate([np.arange(b, b + BLOK) for b in rng.integers(0, len(asal) - BLOK + 1, nb)])[:len(asal)]
               for _ in range(BOOT)]
        # hanya model yang dipasang pada data sebelum asal; naif dan naif musiman tidak dilatih
        terlatih = [m for m in TERLATIH if m in model and m not in ('naif', 'naifmusim')]
        terbaik = min(terlatih, key=lambda m: np.nanmean(mase[m][ok])) if terlatih else None
        for m in model:
            r = {'h': h, 'model': m, 'MASE': np.nanmean(mase[m][ok]), 'sMAPE': np.nanmean(smape[m][ok]),
                 'MASE_idul': np.nanmean(mase[m][ok & idul[None, :]])}
            q = P[m][:, :, h - 1]
            if np.isfinite(q[..., 0]).any():
                r['cakupan80'] = np.mean(((y >= q[..., 0]) & (y <= q[..., 2]))[ok])
            for ref, nama in [('naif', 'd_naif'), (terbaik, 'd_terbaik')]:
                if ref and ref != m and ref in model:
                    dlt = per_asal[m] - per_asal[ref]
                    bs = [np.nanmean(dlt[i]) for i in idx]
                    r[nama] = f'{np.nanmean(dlt):+.3f} [{np.percentile(bs, 2.5):+.3f}; {np.percentile(bs, 97.5):+.3f}]'
            baris.append(r)
        print(f'pembanding terlatih terbaik (ETS, ARIMA, LightGBM) pada h = {h}: {terbaik}')
        t = pd.DataFrame([b for b in baris if b['h'] == h]).drop(columns='h')
        print(t.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
        # per komoditas
        pk = pd.DataFrame({m: [np.nanmean(mase[m][(kom == k)[:, None] & ok]) for k in np.unique(kom)] for m in model},
                          index=np.unique(kom))
        print(f'MASE per komoditas, h = {h}:')
        print(pk.to_string(float_format=lambda v: f'{v:.3f}'))
    pd.DataFrame(baris).to_csv(os.path.join(T, 'metrik.csv'), index=False)


if __name__ == '__main__':
    main()
