# F1 Podium Prediction

Pipeline untuk mengolah data historis F1, melatih model klasifikasi XGBoost, dan
menyajikan prediksi podium melalui Flask API. Fitur model tetap:

- `grid_position`
- `quali_time_gap`
- `constructor_win_rate_at_circuit`
- `driver_recent_points_avg`

Target: `is_podium`. Pemisahan evaluasi tetap temporal, dengan dua musim
terakhir sebagai test set. Feature engineering hanya menggunakan informasi
yang tersedia sebelum balapan; win rate konstruktor di-lag per balapan agar
hasil balapan yang sama tidak bocor ke fitur.

## Struktur folder

```text
Project_F1_Prediksi/
├── app.py
├── preprocess.py
├── train.py
├── requirements.txt
├── README.md
├── dataset/
│   ├── results.csv
│   ├── races.csv
│   ├── qualifying.csv
│   ├── driver_standings.csv
│   ├── constructors.csv
│   ├── circuits.csv
│   ├── drivers.csv
│   └── f1_processed_dataset.csv
└── f1_model_outputs/
    ├── f1_podium_model.pkl
    ├── feature_meta.json
    └── evaluation_metrics.json
```

Script menggunakan path relatif terhadap direktori file. Folder
`dataset/f1_model_outputs/` yang mungkin ada dari versi lama tidak digunakan;
artefak model kanonis berada di `f1_model_outputs/`.

## Instalasi dan menjalankan lokal

Gunakan Python 3.10–3.13, lalu dari direktori `Project_F1_Prediksi`:

```bash
pip install -r requirements.txt
python preprocess.py
python train.py
python app.py
```

`preprocess.py` membaca CSV mentah di `dataset/` dan menulis
`dataset/f1_processed_dataset.csv`. `train.py` melatih XGBoost menggunakan
dataset tersebut, mengevaluasinya pada dua musim terakhir, lalu menyimpan
model, metadata fitur, dan metrik ke `f1_model_outputs/`.

API hanya memuat model dan dataset saat proses web dimulai; request tidak
menjalankan preprocessing atau training. Bila artefak bawaan masih berasal
dari notebook lama, jalankan preprocessing dan training sekali sebelum
deploy agar model dibangun dari fitur bebas leakage yang dihasilkan pipeline
ini. Deploy juga dapat memakai artefak terlatih yang sudah tersedia.

## API

### Cek status

```http
GET /
GET /health
```

### Prediksi untuk balapan

```http
GET /predict?race_id=1182
```

Contoh dengan `curl`:

```bash
curl "http://localhost:10000/predict?race_id=1182"
```

Contoh response (daftar di bawah diringkas; API mengembalikan semua pembalap
untuk balapan tersebut):

```json
{
  "race_id": 1182,
  "race_name": "Spanish Grand Prix",
  "circuit_name": "Madring",
  "year": 2026,
  "round": 14,
  "predictions": [
    {
      "driver_id": 863,
      "driver_name": "Andrea Kimi Antonelli",
      "code": "ANT",
      "constructor_name": "Mercedes",
      "grid_position": 2,
      "podium_probability": 0.9784,
      "actual_is_podium": 1
    },
    {
      "driver_id": 846,
      "driver_name": "Lando Norris",
      "code": "NOR",
      "constructor_name": "McLaren",
      "grid_position": 1,
      "podium_probability": 0.9552,
      "actual_is_podium": 1
    }
  ]
}
```

`race_id` harus ada di `dataset/f1_processed_dataset.csv`. Endpoint ini
mempertahankan perilaku aplikasi lama yang memprediksi peserta dari dataset
tersebut; ia bukan endpoint untuk mengirim susunan grid balapan baru. Field
`actual_is_podium` disertakan bila tersedia di dataset (berguna untuk data
historis), dan bernilai `null` bila tidak tersedia.

Kode status: `400` jika `race_id` bukan integer, `404` jika balapan tidak
ditemukan, dan `422` jika tidak ada baris dengan fitur lengkap.

Jika frontend di-host di domain berbeda, set environment variable
`FRONTEND_ORIGIN` ke origin frontend, misalnya `https://f1-frontend.example.com`.
Default CORS adalah `*` untuk API prediksi read-only.

## Deploy ke Render

1. Push folder proyek beserta CSV, model, dan metadata ke repository Git.
   Jangan mengandalkan filesystem sementara Render sebagai tempat artefak
   training; artefak yang dipakai aplikasi harus tersedia saat deploy.
2. Di Render, buat **New → Web Service** dan hubungkan repository.
3. Jika repository memiliki folder proyek, isi **Root Directory** dengan
   `Project_F1_Prediksi`.
4. Pilih runtime Python 3.12.
5. Isi **Build Command**:

   ```bash
   pip install -r requirements.txt
   ```

6. Isi **Start Command**:

   ```bash
   gunicorn --bind 0.0.0.0:$PORT app:app
   ```

7. (Opsional) Atur `FRONTEND_ORIGIN` di bagian Environment pada Render.
8. Setelah deploy, cek `/health`, lalu panggil
   `/predict?race_id=1182`.

Jalankan `python preprocess.py` lalu `python train.py` sebelum deploy jika
perlu membangun ulang model. Perintah tersebut tidak dijalankan saat startup
atau ketika menerima request.
