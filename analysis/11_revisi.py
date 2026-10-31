"""PIHPS — analisis revisi sesudah telaah panel 30 Okt 2026. Semua keputusan di sini dibuat SESUDAH hasil pertama
dilihat, kecuali aturan dasar MASE; naskah harus menyebutnya.

Perubahan terhadap 05_evaluasi.py:
- Sasaran bersih: tanggal sasaran bukan tanggal tanpa survei (10_kalender.py) dan sel asli, bukan hasil isi maju.
- Bootstrap blok 8 asal (40 hari kerja, lebih panjang dari horizon terjauh), 10.000 ulangan, blok bergerak, persentil.
- Tambahan: galat absolut persen harga (APE), rasio MAE geometrik terhadap naif, pembanding rerata bergerak
  (ma20, ma60, med20, naifma20 = rata-rata naif dan ma20).
- Aturan pemilih walk-forward: hanya memakai data dan galat yang sudah teramati pada asal itu.
- Kelengketan pada horizon yang dinilai, kemiringan dalam komoditas, peringatan kenaikan, Idulfitri per hari raya,
  plasebo, kepekaan per tahun dan tanpa deret nasional.

Pakai: python 11_revisi.py > revisi.log
"""
import os

import numpy as np
import pandas as pd

T = os.path.join(os.environ.get('PIHPS_DATA', 'C:/Data/PIHPS'), 'turunan')
KONTEKS, HORIZON, BOOT, BLOK = 512, [1, 7, 14, 28], 10000, 8
FONDASI = ['chronos2', 'chronosbolt', 'timesfm', 'moirai']
DASAR = ['naif', 'naifmusim', 'ets', 'arima', 'lgbm']
RERATA = ['ma20', 'ma60', 'med20', 'naifma20']
TAMBAHAN = ['ets1', 'lgbml1', 'timesfm_k256']
SYAWAL = pd.to_datetime(['2024-04-10', '2025-03-31', '2026-03-21'])
BERGERAK = ['bawang_merah', 'cabai_merah_keriting', 'cabai_rawit_merah']
LENGKET = ['beras_medium_1', 'gula_pasir_lokal', 'minyak_goreng_curah']
BURNIN = 52  # asal pertama yang dinilai pemilih: sekitar awal 2025; sebelumnya hanya riwayat
JENDELA_Z = 60  # hari survei terakhir untuk mengukur kelengketan pra-asal

Y = np.load(os.path.join(T, 'Y.npy'))
A = np.load(os.path.join(T, 'asli.npy'))
asal = pd.read_csv(os.path.join(T, 'asal.csv')).indeks.to_numpy()
tgl = pd.to_datetime(pd.read_csv(os.path.join(T, 'tanggal.csv')).tanggal)
der = pd.read_csv(os.path.join(T, 'deret.csv'))
kom = der.komoditas.to_numpy()
survei = ~tgl.isin(pd.to_datetime(pd.read_csv(os.path.join(T, 'tanpa_survei.csv')).tanggal)).to_numpy()
n, no = Y.shape[0], len(asal)
tampil = {'chronos2': 'Chronos-2', 'chronosbolt': 'Chronos-Bolt', 'timesfm': 'TimesFM', 'moirai': 'Moirai-2', 'naif': 'Naive',
          'naifmusim': 'SNaive', 'ets': 'ETS', 'arima': 'ARIMA', 'lgbm': 'LightGBM', 'ma20': 'MA20', 'ma60': 'MA60',
          'med20': 'Med20', 'naifma20': 'Naive+MA20', 'ets1': 'ETS-ns', 'lgbml1': 'LightGBM-L1', 'timesfm_k256': 'TimesFM-256'}


def rata_bergerak():
    """Prakiraan datar dari data sampai asal; satu nilai per (deret, asal) berlaku untuk semua horizon."""
    F = {m: np.full((n, no), np.nan) for m in RERATA}
    for k, o in enumerate(asal):
        F['ma20'][:, k] = np.nanmean(Y[:, o - 19:o + 1], axis=1)
        F['ma60'][:, k] = np.nanmean(Y[:, o - 59:o + 1], axis=1)
        F['med20'][:, k] = np.nanmedian(Y[:, o - 19:o + 1], axis=1)
    F['naifma20'] = (F['ma20'] + Y[:, asal]) / 2
    return F


def indeks_boot(seed, ban):
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(ban / BLOK))
    awal = rng.integers(0, ban - BLOK + 1, (BOOT, nb))
    return (awal[:, :, None] + np.arange(BLOK)).reshape(BOOT, -1)[:, :ban]


def ci(per_asal, idx):
    """Rerata dan CI95 atas asal; per_asal boleh memuat NaN (asal tanpa sasaran bersih)."""
    bs = np.nanmean(per_asal[idx], axis=1)
    return np.nanmean(per_asal), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def teks(t, d=3):
    return f'{t[0]:+.{d}f} [{t[1]:+.{d}f}; {t[2]:+.{d}f}]'


def main():
    P = {m: np.load(os.path.join(T, f'pred_{m}.npy'))[:, :, :, 1] for m in FONDASI + DASAR}
    for m in TAMBAHAN:
        f = os.path.join(T, f'pred_{m}.npy')
        if os.path.exists(f):
            P[m] = np.load(f)[:, :, :, 1]
    Q = {m: np.load(os.path.join(T, f'pred_{m}.npy')) for m in FONDASI + ['ets']}
    F = rata_bergerak()
    for m in RERATA:
        P[m] = np.repeat(F[m][:, :, None], 28, axis=2).astype(np.float32)
    semua = [m for m in FONDASI + DASAR + RERATA + TAMBAHAN if m in P]
    skala = np.array([[np.nanmean(np.abs(np.diff(Y[i, o - KONTEKS + 1:o + 1]))) for o in asal] for i in range(n)])
    # kelengketan pra-asal: bagian perubahan nol pada JENDELA_Z hari survei terakhir
    z = np.full((n, no), np.nan)
    hs = np.flatnonzero(survei)
    for k, o in enumerate(asal):
        j = hs[hs <= o][-JENDELA_Z - 1:]
        z[:, k] = np.nanmean(np.where(np.isfinite(np.diff(Y[:, j], axis=1)), np.diff(Y[:, j], axis=1) == 0, np.nan), axis=1)

    print(f'tanggal tanpa survei {int((~survei).sum())} dari {len(survei)}; pada rentang evaluasi '
          f'{int((~survei[asal[0]:asal[-1] + 29]).sum())} dari {asal[-1] + 29 - asal[0]}')
    Z = Y[:, asal[0]:asal[-1] + 29]
    d = np.diff(Z, axis=1)
    s = survei[asal[0] + 1:asal[-1] + 29]
    print(f'bagian perubahan nol: semua tanggal {np.mean(d[np.isfinite(d)] == 0):.3f}; hanya hari survei '
          f'{np.mean(d[:, s][np.isfinite(d[:, s])] == 0):.3f}')
    for k in np.unique(kom):
        x = d[kom == k][:, s]
        print(f'  {k}: {np.mean(x[np.isfinite(x)] == 0) * 100:.1f} %')

    catat, baris_csv = {}, []

    def simpan(h, m, metrik, kelompok, t3):
        baris_csv.append({'h': h, 'model': m, 'metrik': metrik, 'kelompok': kelompok, 'd': t3[0], 'lo': t3[1], 'hi': t3[2]})
        return t3

    for h in HORIZON:
        y = Y[:, asal + h]
        ok_lama = np.isfinite(y) & (skala > 0)
        ok = ok_lama & survei[asal + h][None, :] & A[:, asal + h]
        print(f'\n===== h = {h}: pasangan {int(ok.sum())} dari {int(ok_lama.sum())} '
              f'(asal dengan sasaran tanpa survei {int((~survei[asal + h]).sum())}; sel isi maju {int((ok_lama & ~A[:, asal + h]).sum())})')
        p = {m: P[m][:, :, h - 1] for m in semua if np.isfinite(P[m][:, :, h - 1]).all()}
        ae = {m: np.where(ok, np.abs(y - p[m]), np.nan) for m in p}
        mase = {m: ae[m] / skala for m in p}
        ape = {m: ae[m] / y * 100 for m in p}
        smape = {m: 200 * ae[m] / (np.abs(y) + np.abs(p[m])) for m in p}
        idx = indeks_boot(20261030 + h, no)
        print(f'{"model":14s} {"MASE":>7s} {"MASE lama":>9s} {"APE%":>7s} {"rMAE":>6s}')
        c = np.where(np.sum(np.isfinite(ae['naif']), axis=1) > 0)[0]
        mn = np.nanmean(ae['naif'][c], axis=1)
        for m in p:
            lama = np.nanmean((np.abs(y - p[m]) / skala)[ok_lama])
            rm = np.exp(np.mean(np.log(np.nanmean(ae[m][c], axis=1) / mn)))
            print(f'{tampil[m]:14s} {np.nanmean(mase[m]):7.3f} {lama:9.3f} {np.nanmean(ape[m]):7.3f} {rm:6.3f} sMAPE {np.nanmean(smape[m]):.3f}')
        print('selisih terhadap naif, CI95 blok 8: MASE | APE poin | log rMAE × 100')
        for m in [x for x in p if x != 'naif']:
            dm = np.nanmean(mase[m] - mase['naif'], axis=0)
            da = np.nanmean(ape[m] - ape['naif'], axis=0)
            # rMAE: bobot ulangan per asal; kuadrat terkecil tidak perlu, langsung jumlah galat per deret
            W = np.apply_along_axis(lambda r: np.bincount(r, minlength=no), 1, idx[:2000])
            e1, e0 = np.nan_to_num(ae[m][c]), np.nan_to_num(ae['naif'][c])
            ad = np.isfinite(ae['naif'][c]).astype(float)
            s1, s0, sa = e1 @ W.T, e0 @ W.T, ad @ W.T
            with np.errstate(divide='ignore', invalid='ignore'):
                lg = np.log(s1 / s0)
            bs = np.nanmean(np.where(np.isfinite(lg), lg, np.nan), axis=0) * 100
            t0 = np.mean(np.log(np.nanmean(ae[m][c], axis=1) / mn)) * 100
            print(f'  {tampil[m]:14s} {teks(simpan(h, m, "MASE", "semua", ci(dm, idx)))} | {teks(simpan(h, m, "MAPE", "semua", ci(da, idx)))} | {t0:+.2f} [{np.percentile(bs, 2.5):+.2f}; {np.percentile(bs, 97.5):+.2f}]')
        if 'ets' in p:
            print('selisih terhadap ETS, CI95 blok 8: MASE | APE poin')
            for m in FONDASI + ['naif']:
                dm = np.nanmean(mase[m] - mase['ets'], axis=0)
                da = np.nanmean(ape[m] - ape['ets'], axis=0)
                print(f'  {tampil[m]:14s} {teks(ci(dm, idx))} | {teks(ci(da, idx))}')
        if h > 1:
            print('per komoditas, selisih terhadap naif: APE poin [CI] / MASE [CI]; kolom awal APE naif')
            for k in np.unique(kom):
                b = kom == k
                print(f'  {k} (naif APE {np.nanmean(ape["naif"][b]):.3f}, MASE {np.nanmean(mase["naif"][b]):.3f})')
                for m in ['timesfm', 'moirai', 'chronos2', 'chronosbolt', 'ets', 'ma20', 'naifma20', 'med20', 'ma60']:
                    da = np.nanmean((ape[m] - ape['naif'])[b], axis=0)
                    dm = np.nanmean((mase[m] - mase['naif'])[b], axis=0)
                    print(f'    {tampil[m]:12s} APE {teks(simpan(h, m, "MAPE", k, ci(da, idx)))}  MASE {teks(simpan(h, m, "MASE", k, ci(dm, idx)))}')
            print('TimesFM lawan MA20 (per kelompok), APE poin [CI]')
            for nama, ks in (('bergerak', BERGERAK), ('lengket', LENGKET), ('semua', list(np.unique(kom)))):
                b = np.isin(kom, ks)
                for m in ['timesfm', 'moirai', 'chronos2']:
                    print(f'  {nama} {tampil[m]} - MA20: {teks(ci(np.nanmean((ape[m] - ape["ma20"])[b], axis=0), idx))} ; '
                          f'- Naive+MA20: {teks(ci(np.nanmean((ape[m] - ape["naifma20"])[b], axis=0), idx))}')
        catat[h] = dict(y=y, ok=ok, p=p, ae=ae, mase=mase, ape=ape, idx=idx)
        # kelengketan pada horizon
        if h > 1:
            dh = np.where(ok, np.abs(y / Y[:, asal] - 1) * 100, np.nan)
            print(f'jendela {h} hari: bagian tanpa perubahan bersih (%), median |perubahan| %, per komoditas')
            for k in np.unique(kom):
                b = kom == k
                print(f'  {k}: {np.nanmean(np.where(ok[b], y[b] == Y[:, asal][b], np.nan)) * 100:.1f} ; {np.nanmedian(dh[b]):.2f}')
            flat = np.where(ok, y == Y[:, asal], np.nan) == 1
            print('dekomposisi rugi TimesFM - naif pada komoditas lengket: rerata APE poin × bagian jendela')
            b = np.isin(kom, LENGKET)
            dl = (ape['timesfm'] - ape['naif'])[b]
            fl = flat[b]
            print(f'  jendela datar {np.nanmean(fl[ok[b]]) * 100:.1f} %: selisih {np.nanmean(dl[fl]):+.3f}; '
                  f'jendela bergerak: selisih {np.nanmean(dl[~fl & ok[b]]):+.3f}')

    # kemiringan dalam komoditas: per asal, regresi selisih atas z setelah dikurangi rerata komoditas
    print('\n===== kemiringan selisih terhadap kelengketan pra-asal, dalam komoditas (per 0,1 z), per asal lalu dirata')
    for h in [7, 14, 28]:
        c = catat[h]
        for nama, skor in (('APE', c['ape']), ('MASE', c['mase'])):
            b = np.full(no, np.nan)
            for k in range(no):
                s = pd.DataFrame({'d': (skor['timesfm'] - skor['naif'])[:, k], 'z': z[:, k], 'k': kom}).dropna()
                if len(s) < 50:
                    continue
                s = s[['d', 'z']] - s.groupby('k')[['d', 'z']].transform('mean')
                if s.z.std() > 0:
                    b[k] = (s.d * s.z).sum() / (s.z ** 2).sum() * 0.1
            print(f'h = {h} {nama}: {teks(ci(b, c["idx"]))}')
        s = pd.DataFrame({'d': np.nanmean((c['ape']['timesfm'] - c['ape']['naif']), axis=1), 'z': np.nanmean(z, axis=1), 'k': kom})
        print(f'  antar deret: Spearman z-selisih APE {s.d.corr(s.z, method="spearman"):+.3f}; dalam komoditas '
              + ' '.join(f'{k[:8]}={g.d.corr(g.z, method="spearman"):+.2f}' for k, g in s.groupby('k')))

    # pemilih walk-forward
    print(f'\n===== pemilih walk-forward; dinilai pada asal indeks >= {BURNIN} ({tgl[asal[BURNIN]].date()} dst.)')
    for kand in ['timesfm', 'moirai']:
        for h in [7, 14, 28]:
            c = catat[h]
            lag = int(np.ceil(h / 5))  # asal k' dengan sasaran <= asal k: k' + h/5 <= k
            ms, ap = c['mase'], c['ape']
            m1, a1, tauu = [np.full((n, no), np.nan) for _ in range(3)]
            m2, a2 = np.full((n, no), np.nan), np.full((n, no), np.nan)
            pakai1, pakai2 = np.full((n, no), np.nan), np.full((n, no), np.nan)
            for k in range(BURNIN, no):
                lalu = np.arange(0, k - lag + 1)
                # R1 ambang kelengketan, dipilih pada asal lampau
                grid = np.arange(0.2, 1.01, 0.1)
                bi = [np.nanmean(np.where(z[:, lalu] < t, ms[kand][:, lalu], ms['naif'][:, lalu])) for t in grid]
                t = grid[int(np.nanargmin(bi))]
                tauu[:, k] = t
                c1 = z[:, k] < t
                # R2 unggul pada 26 asal lampau untuk deret itu
                l2 = lalu[-26:]
                c2 = np.nanmean(ms[kand][:, l2], axis=1) < np.nanmean(ms['naif'][:, l2], axis=1)
                m1[:, k], a1[:, k] = np.where(c1, ms[kand][:, k], ms['naif'][:, k]), np.where(c1, ap[kand][:, k], ap['naif'][:, k])
                m2[:, k], a2[:, k] = np.where(c2, ms[kand][:, k], ms['naif'][:, k]), np.where(c2, ap[kand][:, k], ap['naif'][:, k])
                pakai1[:, k], pakai2[:, k] = c1, c2
            ev = np.arange(BURNIN, no)
            idx = indeks_boot(20261031 + h, len(ev))
            pd_m = lambda a, b: np.nanmean(a[:, ev] - b[:, ev], axis=0)
            mavg = np.where(c['ok'], np.abs(c['y'] - (c['p'][kand] + c['p']['naif']) / 2) / skala, np.nan)
            aavg = np.where(c['ok'], np.abs(c['y'] - (c['p'][kand] + c['p']['naif']) / 2) / c['y'] * 100, np.nan)
            print(f'\n{tampil[kand]} h = {h}: ambang akhir {np.nanmean(tauu[:, -1]):.1f}; rerata naif MASE {np.nanmean(ms["naif"][:, ev]):.3f} APE {np.nanmean(ap["naif"][:, ev]):.3f}; '
                  f'{tampil[kand]} MASE {np.nanmean(ms[kand][:, ev]):.3f} APE {np.nanmean(ap[kand][:, ev]):.3f}')
            for nama, mm, aa in (('R1 ambang', m1, a1), ('R2 riwayat', m2, a2), ('R3 rata2', mavg, aavg)):
                print(f'  {nama}: MASE {np.nanmean(mm[:, ev]):.3f} APE {np.nanmean(aa[:, ev]):.3f} | lawan naif MASE {teks(ci(pd_m(mm, ms["naif"]), idx))} APE {teks(ci(pd_m(aa, ap["naif"]), idx))}'
                      f' | lawan model MASE {teks(ci(pd_m(mm, ms[kand]), idx))} APE {teks(ci(pd_m(aa, ap[kand]), idx))}')
            if kand == 'timesfm':
                print('  bagian asal memakai model: ' + ' '.join(f'{k[:9]}={np.nanmean(pakai1[kom == k][:, ev]):.2f}/{np.nanmean(pakai2[kom == k][:, ev]):.2f}' for k in np.unique(kom)))
                print('  per komoditas, R1 lawan naif MASE: ' + ' '.join(f'{k[:9]}={np.nanmean((m1 - ms["naif"])[kom == k][:, ev]):+.3f}' for k in np.unique(kom)))

    # peringatan kenaikan
    print('\n===== peringatan kenaikan >= 5 % / 10 % dari harga asal: hit, false alarm, presisi (median prakiraan)')
    for h in [7, 14, 28]:
        c = catat[h]
        y0 = Y[:, asal]
        for ang in (0.05, 0.10):
            ev = (c['y'] / y0 - 1 >= ang)
            for nama, b in (('semua', np.ones(n, bool)), ('bergerak', np.isin(kom, BERGERAK))):
                okb = c['ok'] & b[:, None]
                baris = []
                for m in ['naif', 'timesfm', 'moirai', 'chronos2', 'ets', 'ma20']:
                    al = c['p'][m] / y0 - 1 >= ang
                    hit = (al & ev)[okb].sum() / max(ev[okb].sum(), 1)
                    fa = (al & ~ev)[okb].sum() / max((~ev)[okb].sum(), 1)
                    pr = (al & ev)[okb].sum() / max(al[okb].sum(), 1)
                    baris.append(f'{tampil[m]} {hit:.3f}/{fa:.3f}/{pr:.3f}')
                print(f'h={h} >={ang:.0%} {nama} (kejadian {ev[okb].mean():.3f}): ' + ' | '.join(baris))
        arah = np.sign(c['y'] - Y[:, asal])
        print(f'h={h} arah benar saat harga berubah: ' + ' '.join(
            f'{tampil[m]} {np.mean((np.sign(c["p"][m] - Y[:, asal]) == arah)[c["ok"] & (arah != 0)]):.3f}' for m in ['timesfm', 'moirai', 'chronos2', 'ets']))

    # Idulfitri
    print('\n===== Idulfitri: selisih terhadap naif per hari raya, MASE | APE; plasebo = jendela asal geser di luar hari raya')
    for h in [7, 14, 28]:
        c = catat[h]
        ts = tgl.to_numpy()[asal + h]
        for lebar, nm in ((-30, 'H-30..H+7'), (-45, 'H-45..H+7')):
            mk = np.zeros(no, bool)
            for s in SYAWAL:
                dd = (ts - s.to_datetime64()).astype('timedelta64[D]').astype(int)
                b = (dd >= lebar) & (dd <= 7)
                mk |= b
                print(f'h={h} {nm} {s.date()} ({b.sum()} asal): ' + ' | '.join(
                    f'{tampil[m]} {np.nanmean(c["mase"][m][:, b] - c["mase"]["naif"][:, b]):+.3f}/{np.nanmean(c["ape"][m][:, b] - c["ape"]["naif"][:, b]):+.3f}'
                    for m in ['timesfm', 'moirai', 'chronos2', 'ets', 'lgbm', 'ma20']))
            w = int(round(mk.sum() / 3))
            print(f'  MASE di jendela / di luar: ' + ' '.join(f'{tampil[m]} {np.nanmean(c["mase"][m][:, mk]):.3f}/{np.nanmean(c["mase"][m][:, ~mk]):.3f}' for m in ['naif', 'timesfm', 'moirai', 'ets']))
            for m in ['timesfm', 'moirai']:
                da = np.nanmean(c['ape'][m] - c['ape']['naif'], axis=0)
                roll = pd.Series(da).rolling(w, min_periods=w).mean()
                buka = pd.Series(mk).rolling(w).sum().fillna(1) == 0
                v = roll[buka].dropna()
                print(f'  plasebo {tampil[m]} {nm}: {len(v)} jendela {w} asal, APE persentil 2,5/50/97,5 {v.quantile(.025):+.3f}/{v.median():+.3f}/{v.quantile(.975):+.3f}; hari raya {np.nanmean(da[mk]):+.3f}')

    # kepekaan
    print('\n===== kepekaan: per tahun sasaran dan tanpa deret nasional (selisih terhadap naif, h = 28, CI hanya untuk semua-tahun)')
    c = catat[28]
    th = tgl.dt.year.to_numpy()[asal + 28]
    nas = der.wilayah.str.contains('Semua', case=False).to_numpy()
    print(f'deret nasional: {nas.sum()}')
    for m in ['timesfm', 'moirai', 'chronos2']:
        for y_ in (2024, 2025, 2026):
            b = th == y_
            kel = ' '.join(f'{nm}: MASE {np.nanmean((c["mase"][m] - c["mase"]["naif"])[np.isin(kom, ks)][:, b]):+.3f} APE {np.nanmean((c["ape"][m] - c["ape"]["naif"])[np.isin(kom, ks)][:, b]):+.3f}'
                          for nm, ks in (('bergerak', BERGERAK), ('lengket', LENGKET)))
            print(f'{tampil[m]} {y_} ({b.sum()} asal): semua MASE {np.nanmean(c["mase"][m][:, b] - c["mase"]["naif"][:, b]):+.3f} APE {np.nanmean(c["ape"][m][:, b] - c["ape"]["naif"][:, b]):+.3f}; {kel}')
        print(f'{tampil[m]} provinsi saja: MASE {teks(ci(np.nanmean((c["mase"][m] - c["mase"]["naif"])[~nas], axis=0), c["idx"]))} APE {teks(ci(np.nanmean((c["ape"][m] - c["ape"]["naif"])[~nas], axis=0), c["idx"]))}')

    pd.DataFrame(baris_csv).to_csv(os.path.join(T, 'revisi_selisih.csv'), index=False)
    print('\n===== cakupan 80 % dan lebar relatif, per komoditas (sasaran bersih)')
    for h in [7, 28]:
        c = catat[h]
        for m in ['timesfm', 'moirai', 'ets']:
            q = Q[m][:, :, h - 1]
            ca = (c['y'] >= q[..., 0]) & (c['y'] <= q[..., 2])
            print(f'h={h} {tampil[m]} semua {ca[c["ok"]].mean():.3f}; ' + ' '.join(f'{k[:9]}={ca[(kom == k)[:, None] & c["ok"]].mean():.2f}' for k in np.unique(kom)))


if __name__ == '__main__':
    main()
