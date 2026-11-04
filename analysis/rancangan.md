# Written test designs (rancangan) for scripts 32–35

Each design below was written and committed in the authors' working notes (`Hasil_Uji_PIHPS.md`, a private
repository) before the script it describes was run. The text is copied verbatim, in Indonesian, from the commit
that first recorded it; the commit hash and time are given above each design. Results are in the logs named in
the README, not here.

---

Commit `c2971aa`, 2026-11-03 17:31:12 +0700

## Rancangan skrip 32: cara menangani tanggal tanpa survei mengubah evaluasi? (ditulis 3 Nov 2026, SEBELUM dijalankan)

Latar. Cek pustaka 3 Nov 2026: sedikitnya delapan studi PIHPS harian mengisi tanggal kosong dengan interpolasi linear atau LOCF/isi maju
(Ericko; Oni; Winata; Utari 2025; Mubarak 2023 dan paket `ifpd`; Hidayat 2026; Permana 2026), satu (Syaifuddiin dkk. 2026, JSCE 7(4))
membiarkan hari tanpa survei kosong. Cek data mentah 3 Nov 2026 (`com_14_2025.json`): tabel provinsi GetGridDataKomoditas memuat hari libur
kerja dengan nilai provinsi sama dengan hari sebelumnya (cuti Idulfitri 31 Mar–7 Apr 2025: enam provinsi pertama tetap, baris nasional
bergerak); akhir pekan tidak ada. Jadi data apa adanya sudah berisi salinan di hari libur, setara LOCF.

Pertanyaan. Dengan prakiraan yang sudah ada (tanpa menjalankan ulang model), apakah selisih model − naif dan urutan model berubah bila
sasaran dinilai seperti yang dilakukan studi lain?

Skenario sasaran (asal, konteks, prakiraan, naif, dan skala MASE sama di semua skenario; hanya sasaran yang dinilai berbeda):
- S0 bersih (cara naskah): sasaran hanya tanggal survei dan sel asli (`survei & A`).
- S1 apa adanya: semua tanggal tabel, nilai seperti terbit (salinan di hari libur, ditambah isi maju <= 3 hari dari 02_siapkan).
- S2 interpolasi: nilai di tanggal tanpa survei dan sel bukan asli diganti interpolasi linear (indeks hari kerja) antara tanggal survei
  terdekat dengan sel asli di kiri dan kanan, per deret; semua tanggal dinilai.

Model: Chronos-2, Chronos-Bolt, TimesFM 2.5, Moirai-2, LightGBM-L1, ETS, ARIMA (median), dan naif. 142 asal 2024–2026, h 1, 7, 14, 28.
Keluaran per skenario: MASE dan MAPE rata-rata; selisih model − naif dengan CI95 (bootstrap blok asal seperti skrip 31, benih 20263300 + h);
perubahan selisih S1 − S0 dan S2 − S0 dengan CI95 (pasangan, indeks bootstrap sama); urutan delapan model menurut MASE; bagian sasaran
di tanggal tanpa survei; bagian jendela tak berubah (y = c).

Prediksi (ditulis sebelum menjalankan):
- P1: di h 1, S1 menggeser selisih model − naif ke arah naif (naik) untuk ketujuh model, karena di tanggal salinan naif tepat.
- P2: di h 1, geseran S2 lebih kecil daripada S1 untuk sebagian besar model.
- P3: di h 28, semua geseran (S1 dan S2) kurang dari 0,05 MASE.

Aturan keputusan untuk naskah (dikunci sekarang): dampak disebut **berarti** bila, untuk sedikitnya satu model pada satu horizon,
(a) perubahan S1 − S0 atau S2 − S0 punya CI95 yang tidak memuat nol DAN besarnya >= 0,05 MASE atau >= 0,05 poin MAPE, atau
(b) tanda selisih model − naif berbalik dengan CI95 skenario itu tidak memuat nol, atau (c) tiga besar urutan MASE berubah anggota.
Bila tidak satu pun, dampak disebut **tidak berarti**: sudut "penanganan tanggal tanpa survei" dibuang dari kebaruan, naskah hanya
memuat satu kalimat bahwa evaluasi tahan terhadap cara pengisian. Batas: model tidak dijalankan ulang dengan konteks yang diisi berbeda,
dan penyisipan akhir pekan (kalender penuh) tidak diuji karena horizon model dihitung dalam hari kerja.
---

Commit `a413d80`, 2026-11-03 21:18:32 +0700

## Rancangan skrip 33: peringatan dini lonjakan harga dari kuantil model fondasi (ditulis 3 Nov 2026, SEBELUM prakiraan desil dibuat dan sebelum data uji dilihat untuk tugas ini)

Latar. Pengguna 3 Nov 2026: cari celah lain, naskah IJCCS sekarang dipertahankan sampai ada yang lebih baik. Cek pustaka 3 Nov 2026
(tiga penelusuran, Rujukan di laporan sesi): prediksi lonjakan PIHPS sudah ada dengan model terlatih saja — Handrian dkk. 2026 (ICORE 8:490–505,
doi 10.32424/hq6bh188, teks penuh dibaca: 6 provinsi, label z-score bergulir 90 hari, horizon 14 hari, LSTM/PatchTST/Crossformer/iTransformer,
AUPRC terbaik 0,797, tanpa pembanding persistensi/momentum, 2024 dibuang; batasan mereka: "six provinces under-represent the logistically
complex eastern archipelago"), Kusumawardhani & Hendrawan 2026 (JIPI 11(2), Semarang, SVR/XGBoost, status SIAGA 5–10 %, WASPADA > 10 %),
Putra 2026 (METHODIKA 12(2), Random Forest, presisi 9 %). Tidak ditemukan pemakaian sebaran prakiraan model fondasi zero-shot sebagai peluang
lonjakan harga; Huang dkk. 2026 (arXiv 2607.07951, PM2.5) memakai median saja dan model fondasi kalah dari BiLSTM. Cek data latih saja
(asal_pra, sasaran < indeks 1548): kenaikan >= 20 % dalam 14 hari kerja terjadi pada 6,1 % pasangan (cabai 19–24 %, beras 0,0 %);
>= 10 % pada 12,5 %. Data uji 2024–2026 belum dihitung untuk tugas ini.

Pertanyaan. Apakah peluang lonjakan yang dibaca dari kuantil model fondasi zero-shot memberi peringatan lebih baik daripada klimatologi
per komoditas, aturan momentum, dan pengklasifikasi LightGBM terlatih, di 34 provinsi? Apakah peluang itu menambah informasi bila
dimasukkan ke LightGBM? Bagaimana cakupan selang 80 % model di jendela datar, bergerak, dan lonjakan?

Kejadian. E(i, o, h) = 1 bila Y[i, o + h] >= 1,20 · Y[i, o] (harga asal teramati, sasaran memenuhi `ok` dari G.siapkan: tanggal survei,
sel asli). Utama: h = 14, ambang 20 %. Sekunder: h 7 dan 28; ambang 10 %. Satuan: semua 309 deret, digabung (TPID memantau semua komoditas),
ditambah rincian per komoditas.

Pemisahan. Uji: 142 asal 2024–2026 (asal.csv, sama dengan naskah). Latih untuk pembanding terlatih dan kalibrasi ulang: asal_pra (202 asal,
26 Des 2019 – 20 Nov 2023) dengan sasaran < indeks 1548; LightGBM memakai semua hari kerja sebagai asal dari indeks 512 sampai sasaran < 1548.
Tidak ada angka uji dipakai untuk memilih apa pun.

Prakiraan. Skrip 03 dijalankan ulang dengan PIHPS_KUANTIL=9 (kuantil 0,1…0,9; keluaran pred_<model>_d9.npy dan pred_<model>_pra_d9.npy),
untuk Chronos-2, Chronos-Bolt, TimesFM 2.5, Moirai-2, konteks 512, bobot sama. Median d9 harus sama dengan pred lama (periksa: selisih
maksimum dilaporkan); bila tidak, dilaporkan dan prakiraan lama tetap dipakai naskah sekarang.

Skor peringatan:
- Model fondasi mentah: P(Y > ambang) dari CDF linear sepotong-sepotong pada sembilan kuantil; ekor diperpanjang linear ke CDF 0 di
  q0,1 − (q0,2 − q0,1) dan CDF 1 di q0,9 + (q0,9 − q0,8), dipotong ke [0, 1]. Pemecah seri untuk peringkat: + 1e−6 · q0,9 / ambang.
- Model fondasi terkalibrasi: regresi isotonik skor mentah → kejadian, disetel pada asal_pra per model dan per h (digabung semua komoditas).
- Klimatologi: frekuensi kejadian per komoditas dan h di data latih.
- Momentum: regresi logistik pada boneka komoditas + perubahan log 5 dan 20 hari kerja terakhir + z-score 90 hari (logika label Handrian
  dijadikan aturan), disetel di data latih.
- LightGBM: perubahan log 1…20 hari, perubahan 40 dan 60 hari, z-score 90 hari, bagian hari tak berubah 60 hari, komoditas, provinsi, bulan,
  hari dalam tahun; parameter bawaan, 500 pohon, laju 0,05, benih 20263400. LightGBM + model fondasi: fitur yang sama ditambah skor mentah
  TimesFM 2.5 (asal_pra saja sebagai latih, karena skor model hanya ada di asal itu).

Metrik. Utama: AUPRC (presisi rata-rata) digabung. Lain: Brier skill score terhadap klimatologi; pada ambang alarm yang memaksimalkan F1 di
data latih: presisi, recall, laju alarm palsu; AUPRC per komoditas. Cakupan dan lebar relatif (q0,9 − q0,1)/c selang 80 % per rezim:
datar (Y[o+h] = Y[o]), lonjakan (E = 1), lainnya. Inferensi: bootstrap blok asal (G.boot, blok 8) 2.000 ulangan (AUPRC mahal), benih
20263400 + h, untuk selisih AUPRC dan BSS antarpembanding dengan indeks sama.

Perbandingan utama (dikunci): TimesFM 2.5 terkalibrasi lawan LightGBM, h 14, ambang 20 %, AUPRC gabungan. Model fondasi lain dilaporkan
semua, tanpa memilih yang terbaik sebagai utama.

Prediksi (ditulis sebelum menjalankan):
- P1: kalibrasi ulang menaikkan BSS keempat model fondasi (skor mentah tidak terkalibrasi pada deret lengket).
- P2: TimesFM terkalibrasi mengalahkan momentum dan klimatologi pada AUPRC di h 14 (CI95 tanpa nol).
- P3: LightGBM tidak kalah dari TimesFM terkalibrasi (mengikuti Huang dkk. 2026).
- P4: cakupan selang 80 % > 0,9 di jendela datar dan < 0,5 di jendela lonjakan, untuk keempat model.

Aturan keputusan (dikunci sekarang):
- **Kuat**: selisih AUPRC TimesFM terkalibrasi − LightGBM CI95 > 0, ATAU LightGBM + model fondasi − LightGBM CI95 > 0. Sudut ini menjadi
  naskah baru; naskah IJCCS sekarang tetap sampai naskah baru lolos gerbang.
- **Setara**: CI95 TimesFM terkalibrasi − LightGBM memuat nol DAN TimesFM terkalibrasi − momentum CI95 > 0. Klaim yang boleh: model
  zero-shot menyamai pengklasifikasi terlatih tanpa pelatihan di 34 provinsi; diputuskan bersama pengguna apakah cukup untuk naskah baru.
- **Lemah**: selain itu. Sudut dibuang, naskah IJCCS sekarang dipertahankan; hasil dicatat di sini saja.
Batas: kejadian dinilai di titik horizon, bukan maksimum dalam jendela; asal mingguan saling tumpang tindih (bootstrap blok menangani
sebagian); kuantil di luar 0,1–0,9 tidak ada, jadi ekor diperkirakan; Wu dkk. 2026 (Energies 19(20):4999, kalibrasi Chronos-2/TimesFM pada
harga karbon) baru dibaca abstraknya — klaim kalibrasi menunggu teks penuh.
---

Commit `8256fb6`, 2026-11-03 23:34:15 +0700

## Rancangan skrip 34: uji konfirmasi peringatan lonjakan pada 12 komoditas tak tersentuh (ditulis 3 Nov 2026, SEBELUM prakiraan desil komoditas lain dibuat)

Alasan: pengguna 3 Nov, "saya mau naskah ini yang diperbaiki hingga ready"; hasil skrip 33 (SETARA) akan masuk naskah IJCCS sebagai temuan
kedua, dan satu periode uji pada sembilan komoditas tidak cukup. Data: `C:/Data/PIHPS_lain` (skrip 29: 12 komoditas, 398 deret, indeks
tanggal sama, asal 142 uji dan 202 pra sama). Data uji 2024–2026 belum dilihat untuk tugas peringatan.

Prosedur beku dari skrip 33, tanpa pilihan baru: `33_peringatan.py` dijalankan apa adanya dengan PIHPS_DATA=C:/Data/PIHPS_lain (kejadian,
ambang 20 % dan 10 %, h 7/14/28, fitur dan parameter LightGBM, logistik momentum, klimatologi, isotonik dari asal_pra, ambang F1 dari data
latih, bootstrap blok 2.000, benih). Yang dibekukan adalah prosedur, bukan model: LightGBM, momentum, klimatologi, dan kalibrasi disetel
ulang pada data latih komoditas lain (sasaran < indeks 1548), seperti pengguna nyata akan melakukannya. Prakiraan desil empat model fondasi
dibuat dengan 03 (PIHPS_KUANTIL=9) untuk asal uji dan asal_pra; log ke `fondasi_lain_<model>[_pra]_d9.log`, hasil ke `peringatan_lain.log`.

Hipotesis (dikunci sekarang; utama h 14, ambang 20 %, AUPRC gabungan):
- K1 (utama): TimesFM 2.5 terkalibrasi − momentum, CI95 > 0.
- K2 (utama): TimesFM 2.5 terkalibrasi − LightGBM, batas atas CI95 >= 0 (tidak jelas kalah).
- K3: TimesFM 2.5 terkalibrasi − klimatologi, CI95 > 0.
- K4: cakupan selang 80 % > 0,9 di jendela datar dan < 0,5 di jendela lonjakan, keempat model.
- Dilaporkan tanpa arah: model fondasi lain, h 7 dan 28, ambang 10 %, LightGBM + TimesFM, per komoditas (>= 5 kejadian).

Aturan untuk naskah: **terkonfirmasi** bila K1 dan K2 lulus; naskah boleh menulis bahwa pola SETARA bertahan pada komoditas tak tersentuh.
**Sebagian** bila hanya satu dari K1/K2: naskah melaporkan apa adanya dan klaim peringatan dibatasi pada sembilan komoditas pengembangan.
**Gagal** bila keduanya gagal: klaim peringatan tidak dipakai sebagai temuan utama naskah. Hasil apa pun dilaporkan.
Batas: kejadian di komoditas lain hampir seluruhnya di dua cabai (cabai merah besar, cabai rawit hijau); beras, gula, minyak kemasan
hampir tanpa kejadian, jadi uji ini terutama menguji cabai di provinsi yang sama.

---

Commit `8e1f555`, 2026-11-04 08:29:53 +0700

## Rancangan skrip 35: apakah keunggulan TimesFM atas momentum hanya efek volatilitas atau provinsi? (ditulis 4 Nov 2026, SEBELUM dijalankan)

Alasan: telaah `2026-11-04d-PIHPS-IJCCS`, kritis konsensus (DA X1 + metode X1): momentum dikumpulkan lintas provinsi, tanpa suku
provinsi dan tanpa tingkat volatilitas (z-score membagi volatilitas keluar), padahal peluang lonjakan TimesFM naik dengan lebar selang.
Margin konfirmatori atas momentum bisa jadi efek "deret mana yang bergejolak", yang bisa ditiru aturan per deret tanpa model fondasi.
Pengguna 3 Nov: "saya mau naskah ini yang di perbaiki hingga ready"; 4 Nov: "go". Jujur: data uji kedua set sudah dilihat untuk tugas
peringatan (skrip 33 dan 34), jadi skrip ini bukan uji tak tersentuh; yang dikunci sekarang adalah pembanding, metrik, dan aturan.

Prosedur: semua dari `33_peringatan.py` (kejadian, okmask, fitur, latih sasaran < indeks 1548, isotonik dari asal_pra, bootstrap blok
2.000 dengan indeks dan benih sama), hanya h 14, ambang 20 %, kedua set (sembilan komoditas; 12 komoditas lewat PIHPS_DATA). Pembanding baru,
disetel pada data latih seperti momentum:
- **klimatologi per deret**: laju lonjakan tiap deret (komoditas × wilayah) pada data latih, tanpa penyusutan.
- **momentum + volatilitas**: regresi logistik momentum (boneka komoditas, perubahan log 5 dan 20 hari, z-score 90 hari) ditambah log
  simpangan baku perubahan log harian 60 dan 250 hari terakhir dan logit klimatologi per deret (dipotong ke [1e-4, 1 − 1e-4]).
Metrik baru: **AUPRC di dalam deret** = rerata AUPRC per deret, berbobot jumlah kejadian uji, hanya deret dengan ≥ 1 kejadian dan ≥ 1
bukan-kejadian; bootstrap dengan indeks asal sama. Deskriptif: jumlah episode (rangkaian asal uji berurutan dengan kejadian pada deret yang
sama; dan per komoditas, episode lintas provinsi yang asalnya berselang ≤ 2 asal digabung).

Hipotesis (dikunci sekarang; TimesFM 2.5 terkalibrasi, h 14, 20 %, AUPRC; "lulus" = lulus di KEDUA set):
- V1 (utama): TimesFM − momentum + volatilitas, AUPRC gabungan, CI95 > 0.
- V2: TimesFM − klimatologi per deret, AUPRC gabungan, CI95 > 0.
- V3: TimesFM − momentum + volatilitas, AUPRC di dalam deret, CI95 > 0.
- Dilaporkan tanpa arah: LightGBM dan momentum asli pada kedua metrik; TimesFM mentah.

Aturan untuk naskah: **bertahan** bila V1 lulus: naskah boleh menulis bahwa TimesFM unggul atas aturan murah termasuk yang tahu volatilitas
deret. **Menyempit** bila V1 gagal di salah satu set: klaim konfirmatori atas momentum dilaporkan bersama catatan bahwa aturan murah yang
tahu volatilitas deret tidak jelas tertinggal; judul dan abstrak tidak lagi menyebut TimesFM lebih baik daripada aturan murah tanpa batasan.
V2 dan V3 dilaporkan apa adanya; V3 gagal berarti naskah tidak boleh membaca margin sebagai keterampilan waktu di dalam deret. Hasil apa pun
dilaporkan.
