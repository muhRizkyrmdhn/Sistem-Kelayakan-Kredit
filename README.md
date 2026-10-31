# 💳 Sistem Penilaian Kelayakan Kredit

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

Aplikasi web prediksi kelayakan kredit berbasis **Machine Learning** yang dibangun dengan **Streamlit** dan **scikit-learn**. Sistem ini menggunakan tiga algoritma klasifikasi (Logistic Regression, Naive Bayes, Decision Tree) untuk menghasilkan probabilitas kelayakan kredit dan kalkulasi *expected value* ekonomi secara real-time.

> **Kategori:** Probabilitas Terapan · Pengambilan Keputusan Algoritmik

---

## 🚀 Demo

Klik badge di atas atau buka: `https://share.streamlit.io/<username>/kredit_app/app.py`

---

## ✨ Fitur Utama

- **Prediksi real-time** — masukkan profil pemohon, dapatkan P(layak) seketika
- **3 model ML** — Logistic Regression, Naive Bayes, Decision Tree (bisa dipilih)
- **Kebijakan tiga-zona** — DISETUJUI / TINJAU MANUAL / DITOLAK dengan ambang yang dapat dikonfigurasi
- **Expected Value** — kalkulasi keuntungan/kerugian ekspektasi berbasis probabilitas
- **Kontribusi fitur** — bar chart faktor pendorong keputusan (khusus Logistic Regression)
- **Evaluasi model** — tabel metrik (Akurasi, Presisi, Recall, ROC-AUC) + Confusion Matrix

---

## 📁 Struktur File

```
kredit_app/
├── app.py                  # Aplikasi Streamlit utama
├── generate_data.py        # Generator dataset sintetis
├── dataset_kredit.csv      # Dataset 1.200 rekaman (sudah tersedia)
├── requirements.txt        # Dependensi Python
├── test_app.py             # Smoke test (Streamlit Testing API)
└── README.md
```

---

## 🖥️ Menjalankan Secara Lokal

```bash
# 1. Clone repository
git clone https://github.com/<username>/kredit_app.git
cd kredit_app

# 2. Install dependensi
pip install -r requirements.txt

# 3. (Opsional) Regenerasi dataset
python generate_data.py

# 4. Jalankan aplikasi
streamlit run app.py
```

Buka browser di `http://localhost:8501`

---

## 📦 Dependensi

| Library | Versi Minimum |
|---------|--------------|
| streamlit | ≥ 1.30 |
| scikit-learn | ≥ 1.3 |
| pandas | ≥ 2.0 |
| numpy | ≥ 1.24 |

---

## 🧠 Alur Keputusan

```
Input Pemohon → Pipeline ML (Preprocessing + Classifier)
                         ↓
                    P(layak) ∈ [0,1]
                         ↓
        ┌────────────────┼────────────────┐
        P ≥ t_setuju   between      P < t_tolak
       DISETUJUI    TINJAU MANUAL    DITOLAK
                         ↓
        EV = P(layak)×bunga% − P(gagal)×LGD%
```

---

## 📊 Dataset

Dataset sintetis 1.200 rekaman dibangkitkan via model logistik dengan variabel:
`usia`, `penghasilan_juta`, `tanggungan`, `lama_kerja_tahun`, `riwayat_kredit`, `pinjaman_juta`, `tenor_bulan`, `rasio_cicilan` → label: `layak`

---

## 🧪 Pengujian

```bash
python test_app.py
```

Menjalankan smoke test untuk ketiga model menggunakan Streamlit Testing API.

---

## 📄 Lisensi

MIT License — bebas digunakan untuk tujuan edukasi dan non-komersial.
