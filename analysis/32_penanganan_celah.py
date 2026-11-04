"""PIHPS — apakah cara menangani tanggal tanpa survei mengubah evaluasi? (sasaran S0 bersih, S1 apa adanya, S2 interpolasi)

Rancangan ditulis dan di-commit sebelum dijalankan: Hasil_Uji_PIHPS.md, "Rancangan skrip 32" (commit c2971aa).
Prakiraan, asal, naif, dan skala MASE sama di semua skenario; hanya sasaran yang dinilai berbeda.

Pakai: venv-gpu python 32_penanganan_celah.py > penanganan_celah.log
"""
import importlib.util
import os

import numpy as np
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
s_ = importlib.util.spec_from_file_location('ta', os.path.join(D, '27_telaah_akhir.py'))
K = importlib.util.module_from_spec(s_)
s_.loader.exec_module(K)
G, T = K.G, K.T
Y, A, survei = G.Y.astype(float), G.A, G.survei
HORIZON = [1, 7, 14, 28]
MODEL = [('Chronos-2', 'chronos2'), ('Chronos-Bolt', 'chronosbolt'), ('TimesFM 2.5', 'timesfm'), ('Moirai-2', 'moirai'),
         ('LightGBM-L1', 'lgbml1'), ('ETS', 'ets'), ('ARIMA', 'arima')]
SKEN = ('S0 bersih', 'S1 apa adanya', 'S2 interpolasi')


def interpolasi():
    """Ganti nilai di tanggal tanpa survei dan sel bukan asli dengan interpolasi linear antar tanggal survei bersel asli."""
    Yi = np.full_like(Y, np.nan)
    x = np.arange(Y.shape[1])
    for i in range(Y.shape[0]):
        v = survei & A[i] & np.isfinite(Y[i])
        if v.sum() < 2:
            continue
        Yi[i] = np.interp(x, x[v], Y[i, v], left=np.nan, right=np.nan)
        Yi[i, v] = Y[i, v]
    return Yi


def ci(x, idx):
    b = np.nanmean(x[idx], axis=1)
    return np.nanmean(x), np.percentile(b, 2.5), np.percentile(b, 97.5)


def ci_beda(x1, x0, idx):
    """Selisih rata-rata dua skenario (asal yang dinilai boleh berbeda), indeks bootstrap sama."""
    b = np.nanmean(x1[idx], axis=1) - np.nanmean(x0[idx], axis=1)
    return np.nanmean(x1) - np.nanmean(x0), np.percentile(b, 2.5), np.percentile(b, 97.5)


def teks(t):
    return f'{t[0]:+.3f} [{t[1]:+.3f}; {t[2]:+.3f}]'


def per_asal(e, ok):
    n = ok.sum(0).astype(float)
    n[n == 0] = np.nan
    return np.where(ok, e, 0.0).sum(0) / n


def main():
    uji = pd.read_csv(os.path.join(T, 'asal.csv')).indeks.to_numpy()
    Yi = interpolasi()
    print(f'deret {Y.shape[0]}, asal {len(uji)}; sel diubah oleh interpolasi pada rentang evaluasi: '
          f'{np.mean((Yi != Y)[:, uji[0]:uji[-1] + 29][np.isfinite(Y[:, uji[0]:uji[-1] + 29])]) * 100:.2f} %')
    berarti = []
    for h in HORIZON:
        idx = G.boot(len(uji), 20263300 + h)
        print(f'\n===== h={h}')
        hasil = {}
        for nama, s in [('naif', None)] + MODEL:
            Q = np.load(os.path.join(T, f'pred_{s or "timesfm"}.npy'))
            b = G.siapkan(uji, Q, h)
            c, sk = b['c'], b['skala']
            m = c if s is None else Q[:, :, h - 1, 1].astype(float)
            dasar = np.isfinite(m) & np.isfinite(c) & (sk > 0)
            yy = {SKEN[0]: Y[:, uji + h], SKEN[1]: Y[:, uji + h], SKEN[2]: Yi[:, uji + h]}
            ok = {SKEN[0]: b['ok'] & dasar, SKEN[1]: np.isfinite(Y[:, uji + h]) & dasar,
                  SKEN[2]: np.isfinite(Yi[:, uji + h]) & dasar}
            hasil[nama] = {}
            for sc in SKEN:
                y = yy[sc]
                # selisih terhadap naif per pasangan (naif = c)
                dm = (np.abs(y - m) - np.abs(y - c)) / sk
                dp = (np.abs(y - m) - np.abs(y - c)) / y * 100
                hasil[nama][sc] = dict(mase=per_asal(np.abs(y - m) / sk, ok[sc]), mape=per_asal(np.abs(y - m) / y * 100, ok[sc]),
                                       dm=per_asal(dm, ok[sc]), dp=per_asal(dp, ok[sc]), ok=ok[sc], y=y, c=c)
        b0 = hasil['naif']
        for sc in SKEN:
            okc, y, c = b0[sc]['ok'], b0[sc]['y'], b0[sc]['c']
            jt = np.isin(uji + h, np.flatnonzero(~survei))
            print(f'  {sc}: pasangan {int(okc.sum())}; sasaran di tanggal tanpa survei {np.mean(np.broadcast_to(jt, okc.shape)[okc]) * 100:.1f} %; '
                  f'jendela tak berubah {np.mean((y == c)[okc]) * 100:.1f} %')
        print('  MASE / MAPE rata-rata per skenario (S0 | S1 | S2):')
        for nama in hasil:
            print(f'    {nama:13s} ' + ' | '.join(f'{np.nanmean(hasil[nama][sc]["mase"]):.3f} / {np.nanmean(hasil[nama][sc]["mape"]):.3f}' for sc in SKEN))
        urut = {}
        for sc in SKEN:
            urut[sc] = sorted(hasil, key=lambda k: np.nanmean(hasil[k][sc]['mase']))
            print(f'  urutan MASE {sc}: {", ".join(urut[sc])}')
        if set(urut[SKEN[0]][:3]) != set(urut[SKEN[1]][:3]) or set(urut[SKEN[0]][:3]) != set(urut[SKEN[2]][:3]):
            berarti.append(f'h{h} tiga besar berubah')
        print('  model − naif (MASE; MAPE) per skenario, lalu perubahan S1−S0 dan S2−S0:')
        for nama, _ in MODEL:
            r = hasil[nama]
            t = {sc: (ci(r[sc]['dm'], idx), ci(r[sc]['dp'], idx)) for sc in SKEN}
            print(f'    {nama:13s} ' + ' | '.join(f'{sc[:2]} {teks(t[sc][0])}; {teks(t[sc][1])}' for sc in SKEN))
            for sc in SKEN[1:]:
                gm, gp = ci_beda(r[sc]['dm'], r[SKEN[0]]['dm'], idx), ci_beda(r[sc]['dp'], r[SKEN[0]]['dp'], idx)
                print(f'    {"":13s} geser {sc[:2]}−S0: MASE {teks(gm)}; MAPE {teks(gp)}')
                for g, nm in ((gm, 'MASE'), (gp, 'MAPE')):
                    if (g[1] > 0 or g[2] < 0) and abs(g[0]) >= 0.05:
                        berarti.append(f'h{h} {nama} {sc[:2]} {nm} {g[0]:+.3f}')
                a0, a1 = t[SKEN[0]][0], t[sc][0]
                if np.sign(a0[0]) != np.sign(a1[0]) and (a1[1] > 0 or a1[2] < 0):
                    berarti.append(f'h{h} {nama} {sc[:2]} tanda berbalik')
            print(flush=True)
    print('\n===== aturan keputusan (rancangan): ' + ('BERARTI: ' + '; '.join(berarti) if berarti else 'TIDAK BERARTI'))


if __name__ == '__main__':
    main()
