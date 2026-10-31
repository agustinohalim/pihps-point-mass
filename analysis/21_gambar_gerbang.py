"""PIHPS — Gambar 1 naskah: (a) kalibrasi peluang diam, (b) selisih MASE lawan naif pada h = 28, model saja dan bergerbang.

Membaca gerbang_rincian.json (20_gerbang_rincian.py) dan gerbang_lintas.log (19_gerbang_lintas_model.py).
Pakai: python 21_gambar_gerbang.py
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
BIRU, KELABU = '#2a78d6', '#a3a29c'


def h28():
    t = open(os.path.join(D, 'gerbang_lintas.log'), encoding='utf-8').read()
    t = t[t.index('h=28'):t.index('===== 97')]
    pola = r'  (\S.*)\n\s+saja lawan naif:\s+MASE ([+-][\d.]+) \[([+-][\d.]+); ([+-][\d.]+)\].*\n\s+bergerbang lawan naif:\s*MASE ([+-][\d.]+) \[([+-][\d.]+); ([+-][\d.]+)\]'
    return [(m[0], *map(float, m[1:])) for m in re.findall(pola, t)]


def main():
    k = json.load(open(os.path.join(D, 'gerbang_rincian.json'), encoding='utf-8'))['kalibrasi']
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={'width_ratios': [1, 1.45]})
    a.plot([0, .85], [0, .85], color=KELABU, lw=1, ls='--')
    for h, warna, tanda in SERI:
        a.plot(k[h]['p'], k[h]['f'], marker=tanda, ms=4, lw=1.2, color=warna, label=f'{h} days')
    a.set_xlabel('Predicted probability of no net change')
    a.set_ylabel('Observed frequency')
    a.set_title('(a) Calibration, by decile', loc='left', fontsize=9)
    a.legend(frameon=False, fontsize=8, loc='upper left')
    a.grid(color='#e6e5e0', lw=.6)
    d = h28()
    d.sort(key=lambda r: r[1])
    b.axvline(0, color='#52514e', lw=1, ls='--')
    for i, (nama, s, sl, sh, g, gl, gh) in enumerate(d):
        b.plot([s, g], [i, i], color='#d6d5cf', lw=2, zorder=1)
        b.errorbar(s, i, xerr=[[s - sl], [sh - s]], fmt='o', ms=5, mfc='white', color=KELABU, lw=1, capsize=1.5,
                   label='Alone' if i == 0 else None, zorder=2)
        b.errorbar(g, i, xerr=[[g - gl], [gh - g]], fmt='o', ms=5, color=BIRU, lw=1, capsize=1.5,
                   label='Gated' if i == 0 else None, zorder=3)
    b.set_yticks(range(len(d)), [NAMA.get(r[0], r[0]) for r in d])
    b.set_xlabel('MASE difference from naive, 28 days')
    b.set_title('(b) Each forecaster, alone and gated', loc='left', fontsize=9)
    b.legend(frameon=False, fontsize=8, loc='lower right')
    b.grid(axis='x', color='#e6e5e0', lw=.6)
    fig.tight_layout()
    fig.savefig(os.path.join(D, 'gambar', 'gambar3_gerbang.png'), dpi=300)


if __name__ == '__main__':
    main()
