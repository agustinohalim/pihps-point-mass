"""PIHPS — prakiraan zero-shot model fondasi pada setiap asal (rencana: ../Artikel_PIHPS_Rencana.md, bagian A).

Untuk tiap asal o (indeks hari kerja terakhir yang teramati) dan tiap deret, konteks = 512 hari kerja
Y[o−511 .. o]; model memprakirakan 28 hari kerja ke depan. Tidak ada pelatihan atau penyetelan; bobot
dipakai apa adanya. Celah di konteks diserahkan ke penanganan bawaan tiap model (Chronos: NaN sebagai
nilai hilang; TimesFM: interpolasi linear; Moirai: imputasi rerata kausal).

Model dan lingkungan (Hasil_Uji_PIHPS.md, "Lingkungan model"):
  chronos2     amazon/chronos-2                  venv-gpu (CUDA)
  chronosbolt  amazon/chronos-bolt-base          venv-gpu (CUDA)
  timesfm      google/timesfm-2.5-200m-pytorch   venv-gpu (CUDA)
  moirai       Salesforce/moirai-2.0-R-small     venv-moirai (CPU)

Keluaran: C:/Data/PIHPS/turunan/pred_<model>.npy, bentuk (deret, asal, 28, 3) = kuantil 0,1 / 0,5 / 0,9.

Pakai: <python venv> 03_model_fondasi.py <model> > fondasi_<model>.log
Uji kepekaan konteks (sesudah telaah 30 Okt 2026): PIHPS_KONTEKS=256, keluaran pred_<model>_k256.npy.
Uji periode sebelum 2024 (sesudah telaah 31 Okt 2026): PIHPS_ASAL=pra, asal dari asal_pra.csv, keluaran pred_<model>_pra.npy.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

D = os.environ.get('PIHPS_DATA', 'C:/Data/PIHPS')
T = os.path.join(D, 'turunan')
KONTEKS, H = int(os.environ.get('PIHPS_KONTEKS', 512)), 28
PERIODE = os.environ.get('PIHPS_ASAL', '')
BOBOT = {'chronos2': 'amazon/chronos-2', 'chronosbolt': 'amazon/chronos-bolt-base',
         'timesfm': 'google/timesfm-2.5-200m-pytorch', 'moirai': 'Salesforce/moirai-2.0-R-small'}


def peramal(nama):
    """Kembalikan fungsi: daftar konteks (np.float32) -> array (n, H, 3) kuantil 0,1/0,5/0,9."""
    if nama in ('chronos2', 'chronosbolt'):
        import torch
        from chronos import BaseChronosPipeline
        p = BaseChronosPipeline.from_pretrained(BOBOT[nama], device_map='cuda')

        def f(ctx):
            q, _ = p.predict_quantiles([torch.tensor(c) for c in ctx], prediction_length=H,
                                       quantile_levels=[0.1, 0.5, 0.9])
            q = torch.stack([x.squeeze(0) for x in q]) if isinstance(q, list) else q
            return q.numpy()
        return f
    if nama == 'timesfm':
        import timesfm
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(BOBOT[nama])
        m.compile(timesfm.ForecastConfig(max_context=KONTEKS, max_horizon=32, normalize_inputs=True,
                                         use_continuous_quantile_head=True, fix_quantile_crossing=True,
                                         per_core_batch_size=64))

        def f(ctx):
            _, q = m.forecast(horizon=H, inputs=[c.astype(np.float64) for c in ctx])
            return q[:, :, [1, 5, 9]]  # indeks 0 = rerata, 1..9 = kuantil 0,1..0,9
        return f
    if nama == 'moirai':
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        m = Moirai2Forecast(module=Moirai2Module.from_pretrained(BOBOT[nama]), prediction_length=H,
                            context_length=KONTEKS, target_dim=1, feat_dynamic_real_dim=0,
                            past_feat_dynamic_real_dim=0)
        ql = list(m.module.quantile_levels)
        j = [ql.index(0.1), ql.index(0.5), ql.index(0.9)]

        def f(ctx):
            return m.predict(ctx)[:, j, :].transpose(0, 2, 1)
        return f
    raise SystemExit(f'model tidak dikenal: {nama}')


def main():
    nama = sys.argv[1]
    Y = np.load(os.path.join(T, 'Y.npy'))
    asal = pd.read_csv(os.path.join(T, f'asal{"_" + PERIODE if PERIODE else ""}.csv')).indeks.to_numpy()
    f = peramal(nama)
    out = np.full((Y.shape[0], len(asal), H, 3), np.nan, dtype=np.float32)
    t0 = time.time()
    for k, o in enumerate(asal):
        ctx = [Y[i, o - KONTEKS + 1:o + 1].astype(np.float32) for i in range(Y.shape[0])]
        out[:, k] = f(ctx)
        if k % 20 == 0:
            print(f'{nama}: asal {k + 1}/{len(asal)}, {time.time() - t0:.0f} d', flush=True)
    akhiran = ('' if KONTEKS == 512 else f'_k{KONTEKS}') + (f'_{PERIODE}' if PERIODE else '')
    np.save(os.path.join(T, f'pred_{nama}{akhiran}.npy'), out)
    print(f'{nama}: selesai {time.time() - t0:.0f} d; NaN {int(np.isnan(out).sum())}')


if __name__ == '__main__':
    main()
