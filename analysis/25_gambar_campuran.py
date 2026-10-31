"""PIHPS — Gambar 1 naskah: (a) peluang latih lawan frekuensi amatan per komoditas dan horizon, (b) selisih MASE lawan
naif pada h = 28, peramal saja dan bercampuran. Membaca campuran_rincian.json dan campuran.log (24_campuran_dasar.py).
Pakai: python 25_gambar_campuran.py
"""
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': '#52514e', 'text.color': '#0b0b0b',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.labelcolor': '#52514e'})
SERI = [('7', '#2a78d6', 'o'), ('14', '#eb6834', 's'), ('28', '#1baf7a', '^')]
NAMA = {'Chronos-2 dilatih+kov': 'Chronos-2 fine-tuned + calendar', 'Chronos-2 dilatih': 'Chronos-2 fine-tuned'}
BIRU, KELABU = '#2a78d6', '#7d7c76'


def h28():
    t = open(os.path.join(D, 'campuran.log'), encoding='utf-8').read()
    t = t[t.index('===== h=28'):t.index('===== 97')]
    pola = r'  (\S.*)\n\s+saja lawan naif:\s+MASE ([+-][\d.]+) \[([+-][\d.]+); ([+-][\d.]+)\].*\n\s+campuran lawan naif:\s*MASE ([+-][\d.]+) \[([+-][\d.]+); ([+-][\d.]+)\]'
    return [(m[0], *map(float, m[1:])) for m in re.findall(pola, t)]


def main():
    k = json.load(open(os.path.join(D, 'campuran_rincian.json'), encoding='utf-8'))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={'width_ratios': [1, 1.5]})
    a.plot([0, .75], [0, .75], color='#a3a29c', lw=1, ls='--')
    for h, warna, tanda in SERI:
        x = [k[h]['latih'][c] for c in k[h]['latih']]
        y = [k[h]['uji'][c] for c in k[h]['latih']]
        a.scatter(x, y, s=16, marker=tanda, color=warna, label=f'{h} days', zorder=3)
    a.set_xlabel('Share of unchanged windows, 2019–2023')
    a.set_ylabel('Share, 2024–2026')
    a.set_title('(a) Commodity mass by horizon', loc='left', fontsize=9)
    a.legend(frameon=False, fontsize=8, loc='upper left')
    a.grid(color='#e6e5e0', lw=.6)
    d = h28()
    d.sort(key=lambda r: r[1], reverse=True)
    b.axvline(0, color='#52514e', lw=1, ls='--')
    for i, (nama, s, sl, sh, g, gl, gh) in enumerate(d):
        b.errorbar(s, i + .17, xerr=[[s - sl], [sh - s]], fmt='o', ms=4.5, mfc='white', color=KELABU, lw=1, capsize=1.5,
                   label='Alone' if i == 0 else None)
        b.errorbar(g, i - .17, xerr=[[g - gl], [gh - g]], fmt='o', ms=4.5, color=BIRU, lw=1, capsize=1.5,
                   label='Mixed' if i == 0 else None)
    b.set_yticks(range(len(d)), [NAMA.get(r[0], r[0]) for r in d])
    b.set_xlabel('MASE difference from naive, 28 days')
    b.set_title('(b) Each forecaster, alone and mixed', loc='left', fontsize=9)
    b.legend(frameon=False, fontsize=8, loc='upper right')
    b.grid(axis='x', color='#e6e5e0', lw=.6)
    fig.tight_layout()
    fig.savefig(os.path.join(D, 'gambar', 'gambar3_campuran.png'), dpi=300)


if __name__ == '__main__':
    main()
