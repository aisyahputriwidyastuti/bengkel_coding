**KESIMPULAN**

tahap 1 
1. Dataset terdiri dari sejumlah fitur numerik dan kategori dengan total baris dan kolom sesuai hasil .info().
2. Tidak ditemukan missing values pada dataset.
3. Sebagian fitur memiliki jumlah nilai unik yang rendah, menunjukkan adanya data kategorikal.
4. Tidak terdapat data duplikat berdasarkan hasil .duplicated().
5. Distribusi kelas target (NObeyesdad) **tidak seimbang**, artinya beberapa kelas lebih dominan daripada yang lain.
6. Beberapa fitur numerik menunjukkan adanya **outlier**, terutama pada fitur seperti 'Weight' dan 'Height', yang terlihat pada boxplot.
7. Distribusi fitur numerik bervariasi, ada yang mendekati normal, ada pula yang skewed.

tahap 2
1. Missing values ditampilkan, lalu dihapus dari dataset.
2. Duplikasi data dideteksi dan dihapus.
3. Outlier dihapus dari kolom numerik menggunakan metode IQR (Interquartile Range).
4. Fitur kategorikal (kecuali target) diubah menjadi angka menggunakan LabelEncoder.
5. Target label NObeyesdad juga di-encode menjadi numerik.
6. Ketidakseimbangan kelas ditangani dengan metode SMOTE untuk oversampling.
7. Data fitur (X) dinormalisasi menggunakan StandardScaler.
8. Data hasil preprocessing digabung menjadi satu DataFrame akhir (processed_df).
9. Dataset akhir siap digunakan untuk tahap modeling machine learning.
