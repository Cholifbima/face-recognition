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

### Untuk GitHub Push (Folder `src/`)
```
src/                        # 📁 Folder utama untuk push ke GitHub
├── main.py                # 🖥️ GUI aplikasi utama
├── eigenface_engine.py    # 🧠 Core algoritma eigenface manual
├── utils.py               # 🔧 Fungsi utility
├── requirements.txt       # 📦 Dependencies Python
├── dataset/               # 📷 Dataset demo (3 celebriti, 590 gambar)
│   ├── Adriana Lima/      
│   ├── Alexandra Daddario/
│   └── Alex Lawther/
├── README.md              # 📖 Dokumentasi lengkap
└── .gitignore            # 🚫 Git ignore file
```

### File Lokal (TIDAK di-push ke GitHub)
```
📁 Root Project/
├── src/                   # 👆 Semua file di atas (untuk GitHub)
├── full_dataset/          # 📷 Dataset lengkap (10 celebriti, 1798 gambar)
├── Laporan/              # 📄 Dokumen laporan Word
└── .venv/                # 🐍 Virtual environment Python lokal
```

## 🐍 Tentang Virtual Environment (.venv)

**`.venv` adalah virtual environment lokal Python** yang berisi:
- Semua package dependencies (numpy, opencv, matplotlib, dll)
- Isolated environment terpisah dari Python global system
- **TIDAK perlu di-push ke GitHub** karena:
  - Ukuran sangat besar (ratusan MB)
  - Specific ke OS dan sistem lokal
  - Bisa di-recreate dengan `pip install -r requirements.txt`

## 🚀 Cara Setup & Menjalankan

### 1. Clone atau Download Project
```bash
# Download folder src/ ke komputer lokal
# Atau clone dari GitHub repository
```

### 2. Setup Virtual Environment (Opsional tapi Recommended)
```bash
# Buat virtual environment baru
python -m venv .venv

# Aktifkan virtual environment
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install semua package yang diperlukan
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi
```bash
# Masuk ke folder src dan jalankan
python main.py
```

## 💻 Cara Menggunakan

### Step 1: Pilih Dataset
1. Buka aplikasi dengan `python main.py`
2. Klik **"Pilih Folder Dataset"**
3. Pilih folder `dataset` (3 orang, included di GitHub)
4. Atau gunakan `full_dataset` jika ada (10 orang, lokal saja)

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

### Dataset Demo (`dataset/` - Included di GitHub)
- **3 Celebriti**: Adriana Lima, Alexandra Daddario, Alex Lawther
- **590 Gambar total**
- **Ukuran**: ~50MB (reasonable untuk GitHub)
- **Untuk demo dan pengumpulan tugas**

### Dataset Lengkap (`full_dataset/` - Lokal Only)
- **10 Celebriti** terkenal
- **1798 Gambar total**
- **Ukuran**: ~200MB (terlalu besar untuk GitHub)
- **Untuk testing advanced**

### Sumber Dataset
- **PINS Face Recognition** dari Kaggle
- Link: https://www.kaggle.com/datasets/hereisburak/pins-face-recognition

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

### File untuk Push ke GitHub (Folder `src/`):
- ✅ `main.py` - Aplikasi GUI utama
- ✅ `eigenface_engine.py` - Implementasi algoritma manual
- ✅ `utils.py` - Fungsi pendukung
- ✅ `dataset/` - Dataset demo (3 celebriti)
- ✅ `requirements.txt` - Dependencies
- ✅ `README.md` - Dokumentasi lengkap
- ✅ `.gitignore` - Git ignore file

### Cara Setup untuk Reviewer:
1. Download/clone folder `src/`
2. `pip install -r requirements.txt`
3. `python main.py`
4. Pilih folder `dataset`
5. Train model dan test recognition

### Cara Demo:
1. Jalankan `python main.py`
2. Pilih folder `dataset` (sudah included)
3. Train model (2-3 menit)
4. Test dengan gambar dari dataset
5. Tunjukkan hasil recognition dengan distance

**✨ Project siap di-push ke GitHub! ✨**