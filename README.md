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

tahap 3 
Pemodelan :
1. Model yang digunakan dan telah dilakukan perbandingan : KNN, LogisticRegression, Random Forest 
2. Pada tahap pemodelan, dilakukan serangkaian proses pra-pemrosesan data untuk memastikan data yang digunakan berkualitas dan siap digunakan oleh algoritma klasifikasi. Langkah-langkah tersebut mencakup penghapusan data duplikat dan nilai kosong, penanganan outlier dengan pendekatan IQR, konversi variabel kategorikal menjadi numerik melalui encoding, penyeimbangan distribusi kelas menggunakan metode SMOTE, serta normalisasi data menggunakan StandardScaler. Tiga algoritma klasifikasi diterapkan dalam analisis ini, yaitu Logistic Regression, Random Forest, dan K-Nearest Neighbors (KNN). Evaluasi performa masing-masing model dilakukan menggunakan beberapa metrik, termasuk akurasi, presisi, recall, dan F1-score, serta divisualisasikan melalui confusion matrix dan grafik perbandingan performa. Berdasarkan hasil evaluasi, salah satu model menunjukkan kinerja terbaik dalam hal F1-score, yang mencerminkan keseimbangan yang baik antara kemampuan model dalam mengidentifikasi kelas positif dan menghindari kesalahan klasifikasi,ini juga ditunjukkan dengan hasil akumulasi presisi dan recaal yg baik. Secara keseluruhan, analisis ini membantu dalam memahami efektivitas masing-masing algoritma serta menentukan model yang paling sesuai untuk klasifikasi tingkat obesitas.

