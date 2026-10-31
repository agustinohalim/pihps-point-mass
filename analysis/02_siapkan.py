"""PIHPS — pembersihan deret dan daftar asal prakiraan (rencana: ../Artikel_PIHPS_Rencana.md, bagian A).

Aturan dikunci di rencana sebelum model dijalankan:
- kalender = semua tanggal kerja yang muncul di PIHPS (gabungan seluruh deret);
- baris "Dum" dibuang (sampah tabel BI, 4 sel 15 Des 2021);
- deret dengan > 10 % tanggal kosong dikeluarkan dan dilaporkan;
- sisa celah diisi maju paling banyak 3 hari kerja; celah lebih panjang dibiarkan kosong (asal yang
  konteks atau sasarannya memuat kosong tidak dinilai untuk deret itu);
- asal prakiraan: tiap hari kerja ke-5 mulai hari kerja pertama ≥ 2024-01-01 sampai ≤ 2026-09-30;
  horizon 1, 7, 14, 28 hari kerja; konteks 512 hari kerja.

Keluaran di C:/Data/PIHPS/turunan: Y.npy (deret × tanggal, Rp/kg, NaN = kosong), deret.csv (komoditas,
wilayah), tanggal.csv, asal.csv (indeks tanggal asal).

Pakai: python 02_siapkan.py > siapkan.log
"""
import os

import numpy as np
import pandas as pd

D = os.environ.get('PIHPS_DATA', 'C:/Data/PIHPS')
T = os.path.join(D, 'turunan')
CELAH_MAKS, ISI_MAJU, KONTEKS = 0.10, 3, 512
HORIZON = [1, 7, 14, 28]


def main():
    os.makedirs(T, exist_ok=True)
    d = pd.read_csv(os.path.join(D, 'harga.csv'), parse_dates=['tanggal'])
    d = d[d.wilayah != 'Dum']
    lebar = d.pivot_table(index=['komoditas', 'wilayah'], columns='tanggal', values='harga', aggfunc='first')
    tgl = lebar.columns
    kosong = lebar.isna().mean(axis=1)
    keluar = kosong[kosong > CELAH_MAKS]
    print(f'{len(lebar)} deret, {len(tgl)} tanggal ({tgl[0].date()} s.d. {tgl[-1].date()})')
    print(f'dikeluarkan (> {CELAH_MAKS:.0%} kosong): {len(keluar)}')
    for (k, w), v in keluar.items():
        print(f'  {k} / {w}: {v:.1%} kosong')
    lebar = lebar.drop(keluar.index)
    sebelum = int(lebar.isna().sum().sum())
    lebar = lebar.T.ffill(limit=ISI_MAJU).T
    print(f'sisa {len(lebar)} deret; sel kosong {sebelum} sebelum isi maju, {int(lebar.isna().sum().sum())} sesudah')

    awal = np.searchsorted(tgl, pd.Timestamp('2024-01-01'))
    akhir = np.searchsorted(tgl, pd.Timestamp('2026-09-30'), side='right') - 1
    asal = np.arange(awal, akhir + 1, 5)
    asal = asal[asal + max(HORIZON) < len(tgl)]
    assert asal[0] >= KONTEKS
    print(f'asal prakiraan {len(asal)}: {tgl[asal[0]].date()} s.d. {tgl[asal[-1]].date()} '
          f'(horizon terjauh {tgl[asal[-1] + max(HORIZON)].date()})')

    np.save(os.path.join(T, 'Y.npy'), lebar.to_numpy(dtype=np.float32))
    lebar.index.to_frame(index=False).to_csv(os.path.join(T, 'deret.csv'), index=False)
    pd.Series(tgl, name='tanggal').to_csv(os.path.join(T, 'tanggal.csv'), index=False)
    pd.Series(asal, name='indeks').to_csv(os.path.join(T, 'asal.csv'), index=False)
    print(lebar.groupby(level=0).size().to_string())


if __name__ == '__main__':
    main()
