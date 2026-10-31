"""PIHPS — menguji aturan pemilih pada data yang tidak melahirkannya, dan membandingkannya dengan pembanding lebih sederhana.

Rancangan ditulis di Hasil_Uji_PIHPS.md sebelum prakiraan pra-2024 dijalankan. Periode pra = asal_pra.csv (202 asal,
Des 2019 – Nov 2023), periode uji = asal.csv (142 asal 2024–2026). Tiga aturan, semuanya walk-forward:
kelengketan (bagian perubahan nol, 60 hari survei terakhir), volatilitas (rerata |Δ log harga| pada jendela yang sama,
ambang desil lampau), daftar komoditas (MASE gabungan satu komoditas pada asal lampau). Lalu ambang dan daftar dibekukan
dari seluruh pra-2024 dan diterapkan ke periode uji, dan LightGBM-L1 dicoba sebagai kandidat pada periode uji.

Pakai: python 14_uji_pra2024.py > uji_pra2024.log
"""
import os

import numpy as np
import pandas as pd

T = os.path.join(os.environ.get('PIHPS_DATA', 'C:/Data/PIHPS'), 'turunan')
KONTEKS, HORIZON, BOOT, BLOK, BURNIN, JENDELA = 512, [7, 14, 28], 10000, 8, 52, 60
BERGERAK = ['bawang_merah', 'cabai_merah_keriting', 'cabai_rawit_merah']
LENGKET = ['beras_medium_1', 'gula_pasir_lokal', 'minyak_goreng_curah']

Y = np.load(os.path.join(T, 'Y.npy'))
A = np.load(os.path.join(T, 'asli.npy'))
tgl = pd.to_datetime(pd.read_csv(os.path.join(T, 'tanggal.csv')).tanggal)
kom = pd.read_csv(os.path.join(T, 'deret.csv')).komoditas.to_numpy()
survei = ~tgl.isin(pd.to_datetime(pd.read_csv(os.path.join(T, 'tanpa_survei.csv')).tanggal)).to_numpy()
n = Y.shape[0]
hs = np.flatnonzero(survei)


def ciri(asal):
    z, v = np.full((n, len(asal)), np.nan), np.full((n, len(asal)), np.nan)
    for k, o in enumerate(asal):
        j = hs[hs <= o][-JENDELA - 1:]
        d = np.diff(Y[:, j], axis=1)
        z[:, k] = np.nanmean(np.where(np.isfinite(d), d == 0, np.nan), axis=1)
        v[:, k] = np.nanmean(np.abs(np.diff(np.log(Y[:, j]), axis=1)), axis=1)
    return z, v


def galat(asal, kandidat, h):
    naif = Y[:, asal]
    skala = np.array([[np.nanmean(np.abs(np.diff(Y[i, o - KONTEKS + 1:o + 1]))) for o in asal] for i in range(n)])
    y = Y[:, asal + h]
    ok = np.isfinite(y) & (skala > 0) & survei[asal + h][None, :] & A[:, asal + h]
    ae = lambda p: np.where(ok, np.abs(y - p), np.nan)
    p = kandidat[:, :, h - 1, 1] if kandidat.ndim == 4 else kandidat[:, :, h - 1]
    return ae(p) / skala, ae(naif) / skala, ae(p) / y * 100, ae(naif) / y * 100


def boot(ban, seed):
    rng = np.random.default_rng(seed)
    aw = rng.integers(0, ban - BLOK + 1, (BOOT, int(np.ceil(ban / BLOK))))
    return (aw[:, :, None] + np.arange(BLOK)).reshape(BOOT, -1)[:, :ban]


def ci(x, idx):
    b = np.nanmean(x[idx], axis=1)
    return f'{np.nanmean(x):+.3f} [{np.percentile(b, 2.5):+.3f}; {np.percentile(b, 97.5):+.3f}]'


def ambang(fitur, lalu, mm, mn, grid, tanda):
    """Ambang dengan MASE gabungan terendah pada asal lampau; tanda +1: pakai model bila fitur < t."""
    nilai = [np.nanmean(np.where(tanda * fitur[:, lalu] < tanda * t, mm[:, lalu], mn[:, lalu])) for t in grid]
    return grid[int(np.nanargmin(nilai))]


def daftar(lalu, mm, mn):
    c = np.zeros(n, bool)
    for q in np.unique(kom):
        b = kom == q
        c[b] = np.nanmean(mm[b][:, lalu]) < np.nanmean(mn[b][:, lalu])
    return c


def walk(asal, mm, mn, ma, an, z, v, h, nama, seed):
    no, lag = len(asal), int(np.ceil(h / 5))
    gz = np.arange(0.2, 1.01, 0.1)
    R = {r: (np.full((n, no), np.nan), np.full((n, no), np.nan)) for r in ('kelengketan', 'volatilitas', 'daftar')}
    for k in range(BURNIN, no):
        l = np.arange(0, k - lag + 1)
        gv = np.nanquantile(v[:, l], np.arange(0.1, 1.0, 0.1))
        pilih = {'kelengketan': z[:, k] < ambang(z, l, mm, mn, gz, 1), 'volatilitas': v[:, k] > ambang(v, l, mm, mn, gv, -1),
                 'daftar': daftar(l, mm, mn)}
        for r, c in pilih.items():
            R[r][0][:, k], R[r][1][:, k] = np.where(c, mm[:, k], mn[:, k]), np.where(c, ma[:, k], an[:, k])
    ev = np.arange(BURNIN, no)
    idx = boot(len(ev), seed)
    pm = lambda a, b: np.nanmean(a[:, ev] - b[:, ev], axis=0)
    print(f'{nama} h={h}: {len(ev)} asal; naif MASE {np.nanmean(mn[:, ev]):.3f} MAPE {np.nanmean(an[:, ev]):.3f}; '
          f'kandidat lawan naif MASE {ci(pm(mm, mn), idx)} MAPE {ci(pm(ma, an), idx)}')
    for r in R:
        print(f'  {r} lawan naif: MASE {ci(pm(R[r][0], mn), idx)} MAPE {ci(pm(R[r][1], an), idx)}; lawan kandidat: MASE {ci(pm(R[r][0], mm), idx)} MAPE {ci(pm(R[r][1], ma), idx)}')
    print(f'  kelengketan - daftar: MASE {ci(pm(R["kelengketan"][0], R["daftar"][0]), idx)} MAPE {ci(pm(R["kelengketan"][1], R["daftar"][1]), idx)}')
    print(f'  kelengketan - volatilitas: MASE {ci(pm(R["kelengketan"][0], R["volatilitas"][0]), idx)} MAPE {ci(pm(R["kelengketan"][1], R["volatilitas"][1]), idx)}')


def belahan(mm, mn, ma, an, nama, h, seed):
    idx = boot(mm.shape[1], seed)
    pm = lambda a, b, s: np.nanmean(a[s] - b[s], axis=0)
    t = [f'{g} {ci(pm(ma, an, np.isin(kom, s)), idx)}' for g, s in (('bergerak', BERGERAK), ('lengket', LENGKET))]
    print(f'{nama} h={h} belahan MAPE: ' + ' | '.join(t))
    print('  per komoditas MAPE: ' + ' '.join(f'{q[:8]}={np.nanmean((ma - an)[kom == q]):+.3f}' for q in np.unique(kom)))


def main():
    pra = pd.read_csv(os.path.join(T, 'asal_pra.csv')).indeks.to_numpy()
    uji = pd.read_csv(os.path.join(T, 'asal.csv')).indeks.to_numpy()
    print(f'pra {len(pra)} asal {tgl[pra[0]].date()}..{tgl[pra[-1]].date()}; uji {len(uji)} asal {tgl[uji[0]].date()}..{tgl[uji[-1]].date()}')
    zp, vp = ciri(pra)
    zu, vu = ciri(uji)
    print('kelengketan per komoditas, pra / uji: ' + ' '.join(f'{q[:8]}={np.nanmean(zp[kom == q]):.2f}/{np.nanmean(zu[kom == q]):.2f}' for q in np.unique(kom)))
    Pp = {m: np.load(os.path.join(T, f'pred_{m}_pra.npy')) for m in ('timesfm', 'moirai')}
    Pu = {m: np.load(os.path.join(T, f'pred_{m}.npy')) for m in ('timesfm', 'moirai', 'lgbml1')}

    print('\n===== A. periode pra-2024, walk-forward; bentuk aturan tidak diubah')
    for m in Pp:
        for h in HORIZON:
            g = galat(pra, Pp[m], h)
            walk(pra, *g, zp, vp, h, m, 20261100 + h)
            belahan(*g, m, h, 20261150 + h)

    print('\n===== B. ambang dan daftar dibekukan dari seluruh pra-2024, diterapkan ke 142 asal 2024-2026')
    gz = np.arange(0.2, 1.01, 0.1)
    for m in ('timesfm', 'moirai'):
        for h in HORIZON:
            mp, np_, _, _ = galat(pra, Pp[m], h)
            l = np.arange(len(pra))
            t = ambang(zp, l, mp, np_, gz, 1)
            lis = daftar(l, mp, np_)
            mm, mn, ma, an = galat(uji, Pu[m], h)
            idx = boot(len(uji), 20261200 + h)
            pm = lambda a, b: np.nanmean(a - b, axis=0)
            cz = zu < t
            print(f'{m} h={h}: ambang {t:.1f}; daftar pakai model: ' + ' '.join(sorted(q[:8] for q in np.unique(kom) if lis[kom == q].all())))
            for nama, c in (('kelengketan beku', cz), ('daftar beku', np.repeat(lis[:, None], len(uji), axis=1))):
                a, b = np.where(c, mm, mn), np.where(c, ma, an)
                print(f'  {nama}: lawan naif MASE {ci(pm(a, mn), idx)} MAPE {ci(pm(b, an), idx)}; lawan model MASE {ci(pm(a, mm), idx)} MAPE {ci(pm(b, ma), idx)}')

    print('\n===== C. periode 2024-2026, walk-forward, kandidat TimesFM dan LightGBM-L1')
    for m in ('timesfm', 'lgbml1'):
        for h in HORIZON:
            walk(uji, *galat(uji, Pu[m], h), zu, vu, h, m, 20261300 + h)


if __name__ == '__main__':
    main()
