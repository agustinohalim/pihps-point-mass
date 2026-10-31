"""PIHPS — Chronos-2 dengan kovariat kalender Idulfitri dan/atau dilatih ulang pada harga sebelum 2024.

Rancangan ditulis di Hasil_Uji_PIHPS.md ("Rancangan opsi 3") sebelum dijalankan.
  kov      zero-shot + kovariat jarak ke Idulfitri       -> pred_chronos2_kov.npy
  latih    dilatih ulang pada harga sebelum 2024         -> pred_chronos2_ft.npy
  latihkov dilatih ulang dengan kovariat, lalu + kovariat -> pred_chronos2_ftkov.npy
  nilai    bandingkan ketiganya dengan Chronos-2 zero-shot dan naif

Pakai: venv-gpu python 17_chronos2_adaptasi.py <mode> > chronos2_<mode>.log
"""
import os
import sys
import time

import numpy as np
import pandas as pd

T = os.path.join(os.environ.get('PIHPS_DATA', 'C:/Data/PIHPS'), 'turunan')
KONTEKS, H = 512, 28
SYAWAL = pd.to_datetime(['2017-06-25', '2018-06-15', '2019-06-05', '2020-05-24', '2021-05-13', '2022-05-02',
                         '2023-04-22', '2024-04-10', '2025-03-31', '2026-03-21', '2027-03-10'])
Y = np.load(os.path.join(T, 'Y.npy'))
tgl = pd.to_datetime(pd.read_csv(os.path.join(T, 'tanggal.csv')).tanggal)
asal = pd.read_csv(os.path.join(T, 'asal.csv')).indeks.to_numpy()
n = Y.shape[0]
d = (tgl.to_numpy()[:, None] - SYAWAL.to_numpy()[None, :]).astype('timedelta64[D]').astype(int)
LEBARAN = np.clip(d[np.arange(len(tgl)), np.abs(d).argmin(axis=1)], -60, 60).astype(np.float32)


def masukan(o, kov):
    out = []
    for i in range(n):
        x = Y[i, o - KONTEKS + 1:o + 1].astype(np.float32)
        if kov:
            out.append({'target': x, 'past_covariates': {'lebaran': LEBARAN[o - KONTEKS + 1:o + 1]},
                        'future_covariates': {'lebaran': LEBARAN[o + 1:o + 1 + H]}})
        else:
            out.append(x)
    return out


def latih(p, kov):
    batas = asal[0]  # semua harga sebelum 1 Jan 2024
    data = []
    for i in range(n):
        x = Y[i, :batas].astype(np.float32)
        data.append({'target': x, 'past_covariates': {'lebaran': LEBARAN[:batas]}, 'future_covariates': {'lebaran': None}}
                    if kov else x)
    t0 = time.time()
    q = p.fit(inputs=data, prediction_length=H, context_length=KONTEKS, learning_rate=1e-6, num_steps=1000,
              batch_size=32, output_dir=os.path.join(T, f'chronos2_ft{"kov" if kov else ""}'), seed=0)
    print(f'pelatihan selesai {time.time() - t0:.0f} d', flush=True)
    return q


def ramal(mode):
    import torch
    from chronos import Chronos2Pipeline
    torch.manual_seed(0)
    p = Chronos2Pipeline.from_pretrained('amazon/chronos-2', device_map='cuda')
    kov = mode in ('kov', 'latihkov')
    if mode.startswith('latih'):
        p = latih(p, kov)
    out = np.full((n, len(asal), H, 3), np.nan, dtype=np.float32)
    t0 = time.time()
    for k, o in enumerate(asal):
        q, _ = p.predict_quantiles(masukan(o, kov), prediction_length=H, quantile_levels=[0.1, 0.5, 0.9])
        out[:, k] = np.stack([x[0].numpy() for x in q])
        if k % 20 == 0:
            print(f'{mode}: asal {k + 1}/{len(asal)}, {time.time() - t0:.0f} d', flush=True)
    akhir = {'kov': 'kov', 'latih': 'ft', 'latihkov': 'ftkov'}[mode]
    np.save(os.path.join(T, f'pred_chronos2_{akhir}.npy'), out)
    print(f'{mode}: selesai; NaN {int(np.isnan(out).sum())}')


def nilai():
    A = np.load(os.path.join(T, 'asli.npy'))
    kom = pd.read_csv(os.path.join(T, 'deret.csv')).komoditas.to_numpy()
    survei = ~tgl.isin(pd.to_datetime(pd.read_csv(os.path.join(T, 'tanpa_survei.csv')).tanggal)).to_numpy()
    skala = np.array([[np.nanmean(np.abs(np.diff(Y[i, o - KONTEKS + 1:o + 1]))) for o in asal] for i in range(n)])
    P = {'zero-shot': 'chronos2', '+kovariat': 'chronos2_kov', 'dilatih': 'chronos2_ft', 'dilatih+kov': 'chronos2_ftkov'}
    P = {k: np.load(os.path.join(T, f'pred_{v}.npy'))[:, :, :, 1] for k, v in P.items() if os.path.exists(os.path.join(T, f'pred_{v}.npy'))}
    no, BOOT, BLOK = len(asal), 10000, 8
    rng = np.random.default_rng(20261500)
    aw = rng.integers(0, no - BLOK + 1, (BOOT, int(np.ceil(no / BLOK))))
    idx = (aw[:, :, None] + np.arange(BLOK)).reshape(BOOT, -1)[:, :no]
    ci = lambda x: f'{np.nanmean(x):+.3f} [{np.percentile(np.nanmean(x[idx], 1), 2.5):+.3f}; {np.percentile(np.nanmean(x[idx], 1), 97.5):+.3f}]'
    for h in (7, 14, 28):
        y, c = Y[:, asal + h], Y[:, asal]
        ok = np.isfinite(y) & (skala > 0) & survei[asal + h][None, :] & A[:, asal + h]
        e = lambda f: np.where(ok, np.abs(y - f), np.nan)
        z = P['zero-shot'][:, :, h - 1]
        print(f'\nh={h}')
        for nama, f in P.items():
            f = f[:, :, h - 1]
            dm, da = np.nanmean((e(f) - e(c)) / skala, 0), np.nanmean((e(f) - e(c)) / y * 100, 0)
            zm, za = np.nanmean((e(f) - e(z)) / skala, 0), np.nanmean((e(f) - e(z)) / y * 100, 0)
            print(f'  {nama:12s} lawan naif MASE {ci(dm)} MAPE {ci(da)} | lawan zero-shot MASE {ci(zm)} MAPE {ci(za)}')
        ts = tgl.to_numpy()[asal + h]
        for s in SYAWAL[7:10]:
            dd = (ts - s.to_datetime64()).astype('timedelta64[D]').astype(int)
            b = (dd >= -30) & (dd <= 7)
            print(f'  Idulfitri {s.date()} ({b.sum()} asal) MAPE lawan naif: ' + ' | '.join(
                f'{k} {np.nanmean((e(f[:, :, h - 1]) - e(c))[:, b] / y[:, b] * 100):+.3f}' for k, f in P.items()))


if __name__ == '__main__':
    nilai() if sys.argv[1] == 'nilai' else ramal(sys.argv[1])
