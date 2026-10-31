"""PIHPS — pembanding yang dilatih pada data sebelum tiap asal (rencana: ../Artikel_PIHPS_Rencana.md, bagian A).

Asal, konteks, dan horizon sama dengan 03_model_fondasi.py (konteks 512 hari kerja, 28 ke depan).
  naif        nilai teramati terakhir (diisi maju bila asal kosong)
  naifmusim   nilai pada hari kerja yang sama minggu lalu: ŷ(o+h) = y(o+h−5·⌈h/5⌉)
  ets         AutoETS statsforecast, musim 5; interval 80 %
  arima       AutoARIMA statsforecast, musim 5; interval 80 %
  ets1, lgbml1  tambahan sesudah telaah 30 Okt 2026: AutoETS tanpa musim; LightGBM dengan objektif L1
  lgbm        LightGBM global atas semua deret, satu model per asal × horizon (1, 7, 14, 28); sasaran
              log(y(t+h)/y(t)); ciri: log rasio lag 1–10, 20, 60, komoditas, wilayah, bulan, dan jarak
              hari kalender tanggal sasaran ke 1 Syawal terdekat (dipotong ±60). Baris latih hanya yang
              sasarannya teramati sampai asal (t + h ≤ o), dari 750 hari kerja terakhir.
statsforecast tidak menerima NaN: konteks diisi maju lalu mundur sebelum dipasang. Pekerja paralel dibatasi
8 (PIHPS_JOBS): n_jobs=-1 membuka 24 proses per asal dan menghabiskan berkas paging Windows.

Tanggal 1 Syawal: 2024–2026 penetapan pemerintah (Hasil_Uji_PIHPS.md); 2018–2023 dari kalender Umm
al-Qura (paket hijridate) — selisih satu hari tidak berarti untuk ciri ini.

Keluaran: C:/Data/PIHPS/turunan/pred_<model>.npy, bentuk (deret, asal, 28, 3) = kuantil 0,1/0,5/0,9
(naif, naifmusim, lgbm: hanya kolom tengah; lgbm hanya horizon 1, 7, 14, 28).

Pakai: venv-gpu python 04_pembanding.py <model> > pembanding_<model>.log
"""
import os
import sys
import time

import numpy as np
import pandas as pd

D = os.environ.get('PIHPS_DATA', 'C:/Data/PIHPS')
T = os.path.join(D, 'turunan')
KONTEKS, H, HORIZON = 512, 28, [1, 7, 14, 28]
LAG = list(range(1, 11)) + [20, 60]
SYAWAL_PEMERINTAH = ['2024-04-10', '2025-03-31', '2026-03-21']


def muat():
    Y = np.load(os.path.join(T, 'Y.npy'))
    asal = pd.read_csv(os.path.join(T, 'asal.csv')).indeks.to_numpy()
    tgl = pd.to_datetime(pd.read_csv(os.path.join(T, 'tanggal.csv')).tanggal)
    deret = pd.read_csv(os.path.join(T, 'deret.csv'))
    return Y, asal, tgl, deret


def isi(x):
    return pd.Series(x).ffill().bfill().to_numpy()


def naif(Y, asal, model):
    out = np.full((Y.shape[0], len(asal), H, 3), np.nan, dtype=np.float32)
    for k, o in enumerate(asal):
        for i in range(Y.shape[0]):
            c = isi(Y[i, o - KONTEKS + 1:o + 1])
            if model == 'naif':
                out[i, k, :, 1] = c[-1]
            else:
                h = np.arange(1, H + 1)
                out[i, k, :, 1] = c[len(c) - 1 + h - 5 * np.ceil(h / 5).astype(int)]
    return out


def statistik(Y, asal, model):
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS
    m = {'ets': AutoETS(season_length=5), 'ets1': AutoETS(season_length=1),
         'arima': AutoARIMA(season_length=5)}[model]
    # simpanan sementara tiap 10 asal: proses pernah dihentikan karena memori sistem menipis
    sem = os.path.join(T, f'pred_{model}_sementara.npy')
    out = np.load(sem) if os.path.exists(sem) else np.full((Y.shape[0], len(asal), H, 3), np.nan, dtype=np.float32)
    t0 = time.time()
    for k, o in enumerate(asal):
        if np.isfinite(out[:, k, :, 1]).all():
            continue
        if k % 10 == 0 and k:
            np.save(sem, out)
        n = Y.shape[0]
        df = pd.DataFrame({'unique_id': np.repeat(np.arange(n), KONTEKS),
                           'ds': np.tile(np.arange(KONTEKS), n),
                           'y': np.concatenate([isi(Y[i, o - KONTEKS + 1:o + 1]) for i in range(n)])})
        f = StatsForecast(models=[m], freq=1, n_jobs=int(os.environ.get('PIHPS_JOBS', 8))).forecast(df=df, h=H, level=[80])
        f = f.sort_values(['unique_id', 'ds'])
        kol = [c for c in f.columns if c not in ('unique_id', 'ds')]
        tengah = [c for c in kol if '-lo-' not in c and '-hi-' not in c][0]
        lo, hi = [c for c in kol if '-lo-80' in c][0], [c for c in kol if '-hi-80' in c][0]
        for j, c in enumerate([lo, tengah, hi]):
            out[:, k, :, j] = f[c].to_numpy().reshape(n, H)
        if k % 10 == 0:
            print(f'{model}: asal {k + 1}/{len(asal)}, {time.time() - t0:.0f} d', flush=True)
    return out


def jarak_syawal(tgl):
    import hijridate
    s = [pd.Timestamp(str(hijridate.Hijri(y, 10, 1).to_gregorian())) for y in range(1439, 1445)]
    s = np.array(s + [pd.Timestamp(x) for x in SYAWAL_PEMERINTAH] + [pd.Timestamp('2027-03-10')],
                 dtype='datetime64[D]')
    d = (tgl.to_numpy(dtype='datetime64[D]')[:, None] - s[None, :]).astype(int)
    return np.clip(d[np.arange(len(d)), np.abs(d).argmin(1)], -60, 60)


def lgbm(Y, asal, tgl, deret, objektif='regression'):
    import lightgbm as lgb
    n, L = Y.shape
    lY = np.log(pd.DataFrame(Y.T).ffill(limit=3).to_numpy().T)
    js = jarak_syawal(tgl)  # asal dipilih sehingga t + 28 selalu di dalam kalender (02_siapkan.py)
    bulan = tgl.dt.month.to_numpy()
    kom = deret.komoditas.astype('category').cat.codes.to_numpy()
    wil = deret.wilayah.astype('category').cat.codes.to_numpy()
    out = np.full((n, len(asal), H, 3), np.nan, dtype=np.float32)

    def ciri(t_idx, h):
        """t_idx: indeks waktu asal (array), untuk semua deret; kembalikan matriks ciri (deret × t)."""
        X = [lY[:, t_idx] - lY[:, t_idx - g] for g in LAG]
        X += [np.repeat(kom[:, None], len(t_idx), 1), np.repeat(wil[:, None], len(t_idx), 1),
              np.repeat(bulan[t_idx][None, :], n, 0)]
        X += [np.repeat(js[t_idx + h][None, :], n, 0)]
        return np.stack([x.reshape(-1) for x in X], 1)

    t0 = time.time()
    for k, o in enumerate(asal):
        for h in HORIZON:
            t_lat = np.arange(max(max(LAG), o - 750), o - h + 1)
            X = ciri(t_lat, h)
            y = (lY[:, t_lat + h] - lY[:, t_lat]).reshape(-1)
            ok = np.isfinite(y) & np.isfinite(X[:, :len(LAG)]).all(1)
            m = lgb.LGBMRegressor(objective=objektif, n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0,
                                  verbose=-1)
            m.fit(X[ok], y[ok], categorical_feature=[len(LAG), len(LAG) + 1])
            Xo = ciri(np.array([o]), h)
            dasar = np.exp(lY[:, o])
            out[:, k, h - 1, 1] = dasar * np.exp(m.predict(Xo))
        if k % 10 == 0:
            print(f'lgbm: asal {k + 1}/{len(asal)}, {time.time() - t0:.0f} d', flush=True)
    return out


def main():
    model = sys.argv[1]
    Y, asal, tgl, deret = muat()
    t0 = time.time()
    if model in ('naif', 'naifmusim'):
        out = naif(Y, asal, model)
    elif model in ('ets', 'ets1', 'arima'):
        out = statistik(Y, asal, model)
    elif model == 'lgbm':
        out = lgbm(Y, asal, tgl, deret)
    elif model == 'lgbml1':
        out = lgbm(Y, asal, tgl, deret, objektif='l1')
    else:
        raise SystemExit(f'model tidak dikenal: {model}')
    np.save(os.path.join(T, f'pred_{model}.npy'), out)
    print(f'{model}: selesai {time.time() - t0:.0f} d; NaN kolom tengah {int(np.isnan(out[..., 1]).sum())}')


if __name__ == '__main__':
    main()
