# Synthetic UMKM Dataset Documentation

## Ringkasan
Dataset ini adalah **data sintetis** yang dirancang untuk meniru karakteristik operasional bisnis UMKM secara realistis pada level bulanan. Data dibuat untuk kebutuhan eksplorasi data, pemodelan machine learning, simulasi analitik, dan latihan NLP pada kolom ulasan.

## Latar Belakang
Dalam praktik operasional UMKM, pengambilan keputusan sering dilakukan tanpa dukungan data yang terstruktur dan terintegrasi. Banyak bisnis hanya memantau omzet secara umum, tetapi belum menghubungkan metrik penting lain seperti margin laba bersih, burn rate, retensi pelanggan, kualitas layanan, dan adopsi teknologi digital dalam satu kerangka analisis yang utuh.

Kondisi ini menimbulkan beberapa permasalahan utama:

- Sulit mendeteksi dini risiko defisit ketika biaya operasional tumbuh lebih cepat daripada pendapatan.
- Sulit membedakan apakah pertumbuhan bisnis didorong oleh volume transaksi yang sehat atau hanya kenaikan nilai transaksi sesaat.
- Sulit mengevaluasi dampak kualitas layanan (rating, volatilitas review, latency) terhadap retensi pelanggan.
- Sulit mengukur kontribusi adopsi digital terhadap efisiensi operasional dan profitabilitas.
- Sulit menilai tekanan pasar lokal akibat tingkat kompetisi pada area bisnis.

Di sisi lain, data riil UMKM sering memiliki kendala akses, privasi, kerahasiaan bisnis, dan kualitas pencatatan yang tidak konsisten. Karena itu, dataset sintetis ini disusun sebagai pendekatan praktis untuk menyediakan data yang aman, terkontrol, dan tetap realistis secara statistik agar dapat digunakan dalam eksplorasi analitik, pengembangan model, serta simulasi pengambilan keputusan berbasis data.

## Tujuan Dataset
Dataset ini memodelkan hubungan antar metrik penting bisnis, termasuk:

- Skala ekonomi (pendapatan dan volume transaksi)
- Efisiensi profitabilitas (margin laba bersih dan burn rate)
- Kualitas layanan (rating, volatilitas review, latency)
- Maturitas bisnis (lama operasional)
- Retensi pelanggan (repeat order rate)
- Adopsi teknologi (digital adoption score)
- Tekanan pasar (location competitiveness)

## Struktur Kolom

| Fitur | Tipe Data | Skala/Satuan | Deskripsi Teknis |
|---|---|---|---|
| `ID` | Integer | Bilangan bulat | Identitas unik baris data sintetis. |
| `Monthly_Revenue` | Integer | IDR | Akumulasi nilai transaksi penjualan kotor dalam satu bulan kalender. |
| `Net_Profit_Margin (%)` | Float | Persentase (%) | Rasio laba bersih terhadap total pendapatan setelah seluruh beban operasional. |
| `Burn_Rate_Ratio` | Float | Rasio | Perbandingan total pengeluaran operasional terhadap pendapatan. Nilai `&gt; 1.0` mengindikasikan defisit. |
| `Transaction_Count` | Integer | Frekuensi | Jumlah nota/transaksi unik pada periode observasi. |
| `Avg_Historical_Rating` | Float | Skala 1-5 | Rata-rata skor penilaian pelanggan. |
| `Review_Text` | String | Teks | Umpan balik tekstual sintetis untuk analisis sentimen/NLP. |
| `Review_Volatility` | Float | Indeks | Tingkat fluktuasi/ketidakkonsistenan kualitas (proxy deviasi rating). |
| `Business_Tenure_Months` | Integer | Bulan | Lama operasional bisnis sejak berdiri. |
| `Repeat_Order_Rate (%)` | Float | Persentase (%) | Rasio transaksi pelanggan lama terhadap total transaksi. |
| `Digital_Adoption_Score` | Float | Skala 1-10 | Indeks adopsi teknologi pembayaran, inventaris, dan kanal digital. |
| `Peak_Hour_Latency` | Categorical | `Low`/`Med`/`High` | Kategori waktu tunggu/keterlambatan proses saat jam sibuk. |
| `Location_Competitiveness` | Integer | Jumlah | Densitas kompetitor sejenis pada area geografis tertentu. |

## Logika Sintesis Data
Generator tidak membuat nilai secara acak murni, tetapi memodelkan hubungan yang masuk akal secara bisnis.

1. `Business_Tenure_Months` dan `Location_Competitiveness` dibangkitkan terlebih dahulu sebagai faktor dasar.
2. `Digital_Adoption_Score` dipengaruhi positif oleh maturitas bisnis (dengan noise).
3. `Transaction_Count` dipengaruhi oleh maturitas, adopsi digital, dan kompetisi lokasi.
4. `Monthly_Revenue` dihitung dari `Transaction_Count` x AOV lognormal + noise musiman.
5. `Peak_Hour_Latency` diturunkan dari tekanan volume transaksi, adopsi digital, dan kompetisi.
6. `Burn_Rate_Ratio` memburuk ketika kompetisi dan latency tinggi.
7. `Net_Profit_Margin (%)` berelasi terbalik dengan `Burn_Rate_Ratio`.
8. `Repeat_Order_Rate (%)` meningkat dengan adopsi digital dan maturitas, turun saat kompetisi tinggi.
9. `Review_Volatility` naik ketika latency tinggi dan burn rate buruk.
10. `Avg_Historical_Rating` dipengaruhi positif digital/profitabilitas dan negatif volatilitas/latency tinggi.
11. `Review_Text` dipilih dari template sentimen yang disejajarkan dengan sinyal kualitas (rating, volatilitas, latency), lalu diberi variasi kalimat tambahan.

## Karakteristik Realisme
- Distribusi pendapatan menggunakan lognormal agar mencerminkan skew ekonomi nyata.
- Hubungan antar variabel dibuat konsisten secara kausal-operasional (bukan random independent).
- Terdapat noise untuk menghindari data terlalu sempurna.
- Ada post-adjustment untuk bisnis dengan defisit berat agar rating/retensi lebih realistis.

## Contoh Use Case
- EDA: distribusi revenue, margin, burn rate, segmentasi risiko.
- Machine Learning: prediksi `Net_Profit_Margin (%)`, klasifikasi `Peak_Hour_Latency`.
- NLP: sentiment analysis pada `Review_Text`.
- Simulasi bisnis: evaluasi dampak adopsi digital terhadap retensi dan profitabilitas.

## Batasan Dataset
- Ini **bukan data riil** dan tidak merepresentasikan entitas bisnis tertentu.
- Distribusi dan hubungan variabel tetap merupakan asumsi desain generator.
- Tidak cocok untuk inferensi kausal kebijakan nyata tanpa kalibrasi ke data empiris.
- Template `Review_Text` masih semi-terstruktur, bukan hasil percakapan pengguna asli.

## Panduan Pemakaian Cepat

```python
import pandas as pd

df = pd.read_csv("synthetic_umkm_data.csv")
print(df.shape)
print(df.head())
```

## Reproducibility
Generator menggunakan seed tetap (`SEED = 42`), sehingga data dapat direproduksi selama parameter generator tidak diubah.