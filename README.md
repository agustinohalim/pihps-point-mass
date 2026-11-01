# pihps-point-mass

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23840467.svg)](https://doi.org/10.5281/zenodo.23840467)

Analysis code and aggregate logs for the manuscript *Holding the Last Price: Zero-Shot Foundation
Models and a Point-Mass Correction on Sticky Daily Food Prices* (in preparation).

Daily food prices in Indonesian provinces often stay unchanged, so pretrained time-series foundation
models can lose to the no-change (naive) forecast on flat series. The method adds a point mass at the
last price to a model's quantile distribution and forecasts the median of the mixture. The mass is one
number per commodity and horizon, the share of unchanged windows in past data, and the model is not
retrained. The method is `median_campuran` in `analysis/16_gerbang.py`; the mass is computed in
`analysis/24_campuran_dasar.py`.

**No data are included.** The PIHPS data belong to Bank Indonesia and are publicly viewable at
https://www.bi.go.id/hargapangan. `01_unduh.py` downloads them; the manuscript uses one download made on
28 October 2026, and values may have been revised since. The logs hold aggregate results only.

## Requirements

Python 3.12.10 with NumPy 2.5.4, pandas 2.3.3, SciPy 1.18.1, requests 2.34.2, matplotlib 3.11.2,
LightGBM 4.7.0, statsforecast 2.1.1, PyTorch 2.14.0 (CUDA), transformers 5.21.0, chronos-forecasting 2.3.2,
and timesfm 3.0.2. Moirai-2 runs in a separate CPU environment: NumPy 1.26.4, pandas 2.1.4, PyTorch 2.4.1,
gluonts 0.14.4, uni2ts 2.0.0.

Model weights come from Hugging Face: `amazon/chronos-2`, `amazon/chronos-bolt-base`,
`google/timesfm-2.5-200m-pytorch`, `Salesforce/moirai-2.0-R-small`.

Set `PIHPS_DATA` to a folder for the data (default `C:/Data/PIHPS`). Downloads go to `mentah/` and derived
arrays and forecasts to `turunan/` under it, never into this repository. Random seeds are set in the
scripts.

## Layout

Scripts, logs, and the JSON the figures read sit together in `analysis/`, because the scripts find them
relative to their own folder. Figures are in `analysis/gambar/`. Each script's docstring gives its usage.
Comments and log text are in Indonesian (*asal* = forecast origin, *naif* = naive, *deret* = series,
*campuran* = mixture, *gerbang* = stay-or-move gate, *komoditas* = commodity). Docstrings mention working
notes (`Artikel_PIHPS_Rencana.md`, `Hasil_Uji_PIHPS.md`) that are not part of this repository.

Scripts 16 to 23 come from an earlier version of the method, a trained stay-or-move classifier; the
manuscript reports it as a comparator.

| Script | Log(s) | Purpose |
|---|---|---|
| `01_unduh.py` | `unduh.log` | Download from the PIHPS site |
| `02_siapkan.py` | `siapkan.log` | Cleaning and forecast origins |
| `03_model_fondasi.py` | `fondasi_*.log` | Zero-shot forecasts of the foundation models |
| `04_pembanding.py` | `pembanding_*.log` | Naive, seasonal naive, ETS, ARIMA, LightGBM |
| `05_evaluasi.py` | `evaluasi*.log` | First evaluation, superseded by `11_revisi.py` |
| `06_ci_komoditas.py` | `ci_komoditas.log` | Intervals by commodity, first evaluation |
| `07_kelengketan.py` | `kelengketan.log` | Share of unchanged days by commodity |
| `08_gambar.py` | — | Figures of earlier manuscript versions |
| `09_periksa_telaah.py` | `periksa_telaah.log` | Checks: calendar and rupiah or percent scale |
| `10_kalender.py` | `kalender.log` | Dates without a survey, found from the data |
| `11_revisi.py` | `revisi.log` | Clean targets, block bootstrap, flat baselines, groups of commodities |
| `12_lgbm_l1.py` | `lgbm_l1.log`, `pembanding_lgbml1.log` | LightGBM with the L1 objective |
| `13_daftar_komoditas.py` | `daftar_komoditas.log` | Per-series rule against a fixed commodity list |
| `14_uji_pra2024.py` | `uji_pra2024.log` | Test on 2022–2023 origins |
| `15_bulat50.py` | `bulat50.log` | Does rounding to Rp 50 explain the loss on sticky series |
| `16_gerbang.py` | `gerbang.log` | Stay-or-move classifier, mixture median, conformal intervals |
| `17_chronos2_adaptasi.py` | `chronos2_*.log` | Chronos-2 fine-tuned, with and without a calendar covariate |
| `18_periksa_gerbang.py` | `periksa_gerbang.log` | Classifier: earlier period, ablation, calibration |
| `19_gerbang_lintas_model.py` | `gerbang_lintas.log` | Classifier on each forecaster's quantiles |
| `20_gerbang_rincian.py` | `gerbang_rincian.log`, `gerbang_rincian.json` | Classifier: details |
| `21_gambar_gerbang.py` | — | Classifier figure of an earlier version |
| `22_gerbang_telaah.py` | `gerbang_telaah.log` | Classifier: further tests |
| `23_gerbang_dasar.py` | `gerbang_dasar.log` | Commodity base rate on 2022–2023, and without national series |
| `24_campuran_dasar.py` | `campuran.log`, `campuran_rincian.json` | **Main method and main results** |
| `25_gambar_campuran.py` | — | Fig. 1 of the manuscript |
| `26_periksa_campuran.py` | `periksa_campuran.log` | One-day horizon and sensitivity to the mass |
| `27_telaah_akhir.py` | `telaah_akhir.log`, `fondasi_timesfm_baru.log` | Global mass, tuned shrinkage, dead band; commodity groups; five later origins, 21 September to 19 October 2026 |
| `28_rupiah_selang.py` | `rupiah_selang.log` | Intervals for rupiah differences by commodity group; sMAPE of naive |

## Versions

- 1.2.0: adds `28_rupiah_selang.py` and its log. The method and the numbers from earlier scripts are unchanged.
- 1.1.0: adds `27_telaah_akhir.py` and its logs, and labels two commodities on Fig. 1. The method and the
  numbers from earlier scripts are unchanged.
- 1.0.0: first release.

## Licence

Code: MIT (see `LICENSE`). The licence does not cover the PIHPS data.
