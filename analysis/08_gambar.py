"""PIHPS — Gambar 1 (selisih MASE dan MAPE terhadap naif per horizon, dengan CI95) dan Gambar 2 (selisih terhadap
naif per komoditas pada h = 28, urut bagian hari tanpa perubahan, MASE dan MAPE berdampingan). Membaca
revisi_selisih.csv dari 11_revisi.py; warna dari palet dataviz. Pakai: python 08_gambar.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

T = os.path.join(os.environ.get('PIHPS_DATA', 'C:/Data/PIHPS'), 'turunan')
K = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gambar')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': '#52514e', 'text.color': '#0b0b0b',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.labelcolor': '#52514e'})
NAMA = {'bawang_merah': 'Shallot', 'bawang_putih': 'Garlic', 'beras_medium_1': 'Medium rice',
        'cabai_merah_keriting': 'Curly red chilli', 'cabai_rawit_merah': "Bird's-eye chilli",
        'daging_ayam_ras': 'Broiler chicken', 'gula_pasir_lokal': 'Granulated sugar',
        'minyak_goreng_curah': 'Bulk cooking oil', 'telur_ayam_ras': 'Chicken eggs'}
SERI = [('timesfm', 'TimesFM 2.5', '#2a78d6', 'o'), ('moirai', 'Moirai-2', '#eb6834', 's'),
        ('chronos2', 'Chronos-2', '#1baf7a', '^'), ('chronosbolt', 'Chronos-Bolt', '#eda100', 'D'),
        ('ets', 'ETS', '#6250d6', 'v')]
BIRU, KELABU = '#2a78d6', '#a3a29c'


def baca():
    return pd.read_csv(os.path.join(T, 'revisi_selisih.csv'))


def gambar1():
    c = baca()
    c = c[c.kelompok == 'semua']
    hs = [7, 14, 28]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for ax, metrik, judul in ((axs[0], 'MASE', 'MASE difference from naive'),
                              (axs[1], 'MAPE', 'MAPE difference from naive (points)')):
        ax.axhline(0, color='#52514e', lw=1, ls='--')
        for j, (kode, nama, warna, tanda) in enumerate(SERI):
            s = c[(c.model == kode) & (c.metrik == metrik)].set_index('h')
            x = np.arange(3) + (j - 2) * .13
            d = [s.d[h] for h in hs]
            ax.errorbar(x, d, yerr=[[s.d[h] - s.lo[h] for h in hs], [s.hi[h] - s.d[h] for h in hs]], fmt=tanda, ms=4,
                        color=warna, lw=1.1, capsize=1.5, label=nama)
        ax.set_xticks(range(3), [str(h) for h in hs])
        ax.set_xlabel('Horizon (working days ahead)')
        ax.set_ylabel(judul)
        ax.grid(axis='y', color='#e6e5e0', lw=.6)
    axs[0].legend(frameon=False, fontsize=7.5, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(K, 'gambar1_relatif_naif.png'), dpi=300)


def gambar2():
    c = baca()
    c = c[(c.h == 28) & (c.model == 'timesfm') & (c.kelompok != 'semua')]
    Y = np.load(os.path.join(T, 'Y.npy'))
    asal = pd.read_csv(os.path.join(T, 'asal.csv')).indeks.to_numpy()
    kom = pd.read_csv(os.path.join(T, 'deret.csv')).komoditas.to_numpy()
    tgl = pd.to_datetime(pd.read_csv(os.path.join(T, 'tanggal.csv')).tanggal)
    survei = ~tgl.isin(pd.to_datetime(pd.read_csv(os.path.join(T, 'tanpa_survei.csv')).tanggal)).to_numpy()
    d = np.diff(Y[:, asal[0]:asal[-1] + 29], axis=1)[:, survei[asal[0] + 1:asal[-1] + 29]]
    nol = {k: np.mean(d[kom == k][np.isfinite(d[kom == k])] == 0) * 100 for k in NAMA}
    urut = sorted(NAMA, key=lambda k: nol[k])
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for ax, metrik, judul in ((axs[0], 'MASE', 'MASE difference from naive'),
                              (axs[1], 'MAPE', 'MAPE difference from naive (points)')):
        ax.axvline(0, color='#52514e', lw=1, ls='--')
        s = c[c.metrik == metrik].set_index('kelompok')
        for i, k in enumerate(urut):
            r = s.loc[k]
            warna = BIRU if r.lo > 0 or r.hi < 0 else KELABU
            ax.errorbar(r.d, i, xerr=[[r.d - r.lo], [r.hi - r.d]], fmt='o', ms=4.5, color=warna, lw=1.1, capsize=2)
        ax.set_xlabel(judul)
        ax.grid(axis='x', color='#e6e5e0', lw=.6)
    axs[0].set_yticks(range(len(urut)), [f'{NAMA[k]} ({nol[k]:.0f} %)' for k in urut])
    axs[0].set_ylabel('Commodity (days with no price change)')
    fig.tight_layout()
    fig.savefig(os.path.join(K, 'gambar2_komoditas.png'), dpi=300)


gambar1()
gambar2()
