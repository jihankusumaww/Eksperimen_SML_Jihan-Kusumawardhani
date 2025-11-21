# Insurance Data Preprocessing Pipeline

Proyek ini melakukan preprocessing otomatis pada dataset asuransi menggunakan scikit-learn pipeline.

## Fitur

- ✅ Preprocessing otomatis untuk data numerik dan kategorikal
- ✅ MinMax Scaling untuk fitur numerik
- ✅ One-Hot Encoding untuk fitur kategorikal
- ✅ Split data training dan testing (70:30)
- ✅ Menyimpan pipeline preprocessing untuk reusability
- ✅ GitHub Actions workflow untuk CI/CD

## Struktur Proyek

```
.
├── insurance_raw.csv                          # Dataset mentah
├── run_preprocessing.py                       # Script utama
├── requirements.txt                           # Dependencies Python
├── preprocessing/
│   ├── automate_jkw.py                       # Modul preprocessing
│   ├── preprocessor.joblib                   # Pipeline tersimpan
│   └── insurance_preprocessing/
│       ├── columns.csv                       # Header kolom
│       ├── insurance_train_preprocessed.csv  # Data training
│       └── insurance_test_preprocessed.csv   # Data testing
└── .github/
    └── workflows/
        └── preprocessing.yml                 # GitHub Actions workflow
```

## Cara Menggunakan

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Jalankan Preprocessing

```bash
python run_preprocessing.py
```

## GitHub Actions Workflow

Workflow akan berjalan otomatis saat:
- Push ke branch `main` atau `master`
- Pull request ke branch `main` atau `master`
- Manual trigger melalui GitHub UI (workflow_dispatch)

Hasil preprocessing akan disimpan sebagai artifacts yang dapat diunduh selama 30 hari.

## Output

Setelah preprocessing, akan dihasilkan:
- `preprocessor.joblib`: Pipeline preprocessing yang dapat digunakan kembali
- `insurance_train_preprocessed.csv`: Data training yang sudah diproses
- `insurance_test_preprocessed.csv`: Data testing yang sudah diproses
- `columns.csv`: Header kolom untuk referensi

## Author

**Jihan Kusumawardhani**

## License

MIT License
