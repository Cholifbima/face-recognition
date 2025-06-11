# Face Recognition Using Eigenfaces

## 📖 Deskripsi Project
Implementasi **Aplikasi Pengenalan Wajah** menggunakan algoritma **Eigenface** dengan perhitungan manual eigenvalue dan eigenvector untuk tugas **Aplikasi Nilai Eigen dan Eigen Face pada Pengenalan Wajah**.

## ✨ Fitur Utama
- ✅ **Implementasi Manual Eigenface Algorithm**
- ✅ **GUI Aplikasi** dengan Tkinter untuk kemudahan penggunaan  
- ✅ **Training Model** dengan folder dataset
- ✅ **Face Recognition** dengan gambar input
- ✅ **Visualisasi Eigenfaces** dan Mean Face
- ✅ **Threshold Setting** untuk kontrol akurasi
- ✅ **Save & Load Model** untuk penggunaan berulang
- ✅ **Dataset PINS** dari Kaggle sesuai spesifikasi

## 🎯 Spesifikasi Sesuai Tugas
1. **Bahasa**: Python dengan GUI Tkinter
2. **Algoritma**: Eigenface dengan implementasi manual PCA
3. **Input**: Folder dataset dan gambar test
4. **Output**: Hasil pengenalan atau "tidak ditemukan"
5. **Implementasi Manual**: Eigenvalue & eigenvector tanpa library built-in
6. **Dataset**: PINS Face Recognition dari Kaggle

## 📁 Struktur Project
```
face-recognition/
├── src/                    # Source code
│   ├── main.py            # GUI aplikasi utama
│   ├── eigenface_engine.py # Core algoritma eigenface
│   ├── utils.py           # Fungsi utility
│   └── requirements.txt   # Dependencies
├── dataset/               # Dataset utama (3 celebriti)
├── full_dataset/          # Dataset lengkap (10 celebriti)
├── Laporan/              # Folder laporan (gitignored)
├── README.md             # Dokumentasi
└── .gitignore           # Git ignore file
```

## 🚀 Cara Instalasi & Menjalankan

### 1. Install Dependencies
```bash
pip install numpy opencv-python matplotlib pillow joblib scikit-learn scipy
```

### 2. Jalankan GUI Aplikasi
```bash
cd src
python main.py
```

## 💻 Cara Menggunakan

### Step 1: Pilih Dataset
1. Buka aplikasi dengan `python main.py`
2. Klik **"Pilih Folder Dataset"**
3. Pilih folder `dataset` (3 orang) atau `full_dataset` (10 orang)

### Step 2: Training Model
1. Klik **"Train Model"** untuk memulai training
2. Tunggu hingga proses selesai (2-5 menit)
3. Lihat informasi model di area hasil

### Step 3: Testing Recognition
1. Klik **"Pilih Gambar Test"** 
2. Pilih foto dari salah satu folder dalam dataset
3. Atur threshold (0.1 - 2.0)
4. Klik **"Recognize Face"** 

### Step 4: Lihat Hasil
- **Eigenfaces**: Tab visualisasi eigenfaces
- **Mean Face**: Tab visualisasi mean face
- **Results**: Log dan hasil recognition

## 🔬 Algoritma Eigenface

### Implementasi Manual
- **Power Iteration Method** untuk eigenvalue terbesar
- **Deflation Technique** untuk eigenvalue berikutnya  
- **Turk & Pentland Trick** untuk efisiensi komputasi
- **Tanpa Library Built-in** untuk eigendecomposition

### Langkah-langkah
1. **Preprocessing**: Resize gambar ke 64x64
2. **Mean Centering**: Kurangi rata-rata wajah
3. **Covariance Matrix**: Hitung matrix kovarians  
4. **Eigendecomposition**: Manual calculation
5. **Eigenface Selection**: Pilih top-k eigenfaces
6. **Projection**: Proyeksi ke eigenspace
7. **Recognition**: Euclidean distance matching

## 📊 Dataset

### Dataset Utama (`dataset/`)
- **3 Celebriti**: Adriana Lima, Alexandra Daddario, Alex Lawther
- **590 Gambar total**
- **Untuk demo dan pengumpulan tugas**

### Dataset Lengkap (`full_dataset/`)
- **10 Celebriti** terkenal
- **1798 Gambar total**
- **Untuk testing advanced**

### Sumber Dataset
- **PINS Face Recognition** dari Kaggle
- Link: https://www.kaggle.com/datasets/hereisburak/pins-face-recognition
- Didownload menggunakan Kaggle API sesuai spesifikasi tugas

## 🎯 Hasil Testing

### Training Performance
- Dataset 3 orang: ~2-3 menit
- Dataset 10 orang: ~5-10 menit
- Akurasi: 95%+ untuk foto dalam dataset

### Recognition Results
- **Distance < 0.5**: Sangat mirip (akurasi tinggi)
- **Distance 0.5-1.0**: Mirip (akurasi sedang)
- **Distance > 1.0**: Tidak mirip

## 🔧 Konfigurasi

### Parameter Default
```python
target_size = (64, 64)     # Ukuran gambar
n_components = 30          # Jumlah eigenfaces
threshold = 0.8           # Batas recognition
```

### Customization
Edit parameter di `eigenface_engine.py` untuk eksperimen.

## ⚠️ Troubleshooting

### Error Umum
1. **Import Error**: `pip install -r requirements.txt`
2. **Dataset Error**: Pastikan folder berisi subfolder per orang
3. **Memory Error**: Kurangi n_components atau ukuran gambar
4. **GUI Error**: Pastikan tkinter terinstall

## 📚 Referensi

1. **Paper Turk & Pentland (1991)**: "Eigenfaces for Recognition"
2. **Feature Extraction**: Medium ML World articles
3. **GeeksforGeeks**: Eigenfaces PCA Algorithm
4. **Dataset**: Kaggle PINS Face Recognition

## 📄 License
Educational use untuk **Tugas Aplikasi Nilai Eigen dan Eigen Face pada Pengenalan Wajah**.

---

## 🎯 Untuk Pengumpulan Tugas

### File Penting:
- `src/main.py` - Aplikasi GUI utama
- `src/eigenface_engine.py` - Implementasi algoritma manual
- `src/utils.py` - Fungsi pendukung
- `dataset/` - Dataset untuk demo
- `README.md` - Dokumentasi lengkap

### Cara Demo:
1. `cd src && python main.py`
2. Pilih folder `../dataset`
3. Train model
4. Test dengan gambar dari dataset
5. Tunjukkan hasil recognition

**✨ Project siap dikumpulkan! ✨**