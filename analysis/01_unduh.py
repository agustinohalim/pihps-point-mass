"""PIHPS Nasional (Bank Indonesia) — unduh harga harian pasar tradisional per provinsi (rencana: ../Artikel_PIHPS_Rencana.md).

Endpoint tabel harga situs BI, diuji 28 Okt 2026: GetGridDataKomoditas mengembalikan satu baris per wilayah
("Semua Provinsi" = nasional, lalu tiap provinsi) dan satu kolom per tanggal dd/mm/yyyy (hari kerja saja).
Satu permintaan = satu komoditas × satu tahun; jeda 2 detik; JSON mentah disimpan apa adanya di
C:/Data/PIHPS/mentah (atau PIHPS_DATA) dan tidak diunduh ulang. Hak data pada Bank Indonesia — data
tidak masuk repo; hanya skrip ini dan agregat.

Komoditas dikunci di rencana bagian A (ID dari GetRefCommodityAndCategory, 28 Okt 2026).
Keluaran: harga.csv (tanggal, wilayah, komoditas, harga Rp/kg).

Pakai: python 01_unduh.py > unduh.log
"""
import json
import os
import time
from datetime import date

import pandas as pd
import requests

D = os.environ.get('PIHPS_DATA', 'C:/Data/PIHPS')
M = os.path.join(D, 'mentah')
URL = 'https://www.bi.go.id/hargapangan/WebSite/TabelHarga/GetGridDataKomoditas'
KEPALA = {'User-Agent': 'Mozilla/5.0 (research; Universitas OSO)', 'X-Requested-With': 'XMLHttpRequest',
          'Referer': 'https://www.bi.go.id/hargapangan/TabelHarga/PasarTradisionalKomoditas'}
KOMODITAS = {'com_3': 'beras_medium_1', 'com_11': 'bawang_merah', 'com_12': 'bawang_putih',
             'com_14': 'cabai_merah_keriting', 'com_16': 'cabai_rawit_merah', 'com_7': 'daging_ayam_ras',
             'com_10': 'telur_ayam_ras', 'com_17': 'minyak_goreng_curah', 'com_21': 'gula_pasir_lokal'}
AWAL, AKHIR = date(2018, 1, 1), date(2026, 10, 31)


def ambil(sesi, kom, th):
    f = os.path.join(M, f'{kom}_{th}.json')
    if os.path.exists(f):
        with open(f, encoding='utf-8') as h:
            return json.load(h)
    akhir = min(date(th, 12, 31), AKHIR)
    par = {'price_type_id': 1, 'comcat_id': kom, 'province_id': '', 'regency_id': '', 'showKota': 'false',
           'showPasar': 'false', 'tipe_laporan': 1, 'start_date': f'{th}-01-01', 'end_date': akhir.isoformat()}
    for coba in range(3):
        time.sleep(2)
        try:
            r = sesi.get(URL, params=par, timeout=180)
            r.raise_for_status()
            js = r.json()
            break
        except (requests.RequestException, ValueError) as e:
            print(f'  {kom} {th} gagal ({coba + 1}): {e}', flush=True)
    else:
        raise SystemExit(f'{kom} {th} gagal tiga kali')
    # tulis ke berkas sementara dulu: proses yang terhenti tidak meninggalkan cache setengah jadi
    with open(f + '.tmp', 'w', encoding='utf-8') as h:
        json.dump(js, h)
    os.replace(f + '.tmp', f)
    return js


def harga(v):
    s = str(v).strip().replace(',', '').replace('.', '')
    return float(s) if s.isdigit() and float(s) > 0 else float('nan')


def main():
    os.makedirs(M, exist_ok=True)
    sesi = requests.Session()
    sesi.headers.update(KEPALA)
    baris = []
    for kom, nama in KOMODITAS.items():
        for th in range(AWAL.year, AKHIR.year + 1):
            data = ambil(sesi, kom, th).get('data', [])
            n0 = len(baris)
            for r in data:
                for k, v in r.items():
                    if len(k) == 10 and k[2] == '/' and k[5] == '/':
                        baris.append((pd.Timestamp(int(k[6:]), int(k[3:5]), int(k[:2])), r['name'].strip(), nama, harga(v)))
            print(f'{nama:22} {th}: {len(data)} wilayah, {len(baris) - n0} sel', flush=True)
    d = pd.DataFrame(baris, columns=['tanggal', 'wilayah', 'komoditas', 'harga']).dropna()
    d.to_csv(os.path.join(D, 'harga.csv'), index=False)
    print(f'\n{len(d)} baris; {d.wilayah.nunique()} wilayah; {d.tanggal.nunique()} tanggal '
          f'({d.tanggal.min().date()} s.d. {d.tanggal.max().date()})')
    print(d.groupby('komoditas').agg(baris=('harga', 'size'), wilayah=('wilayah', 'nunique')).to_string())


if __name__ == '__main__':
    main()
