# 🦴 LUMOS — Lumbar Spine Osteoporosis Screening System

> **Klasifikasi Multimodal Osteoporosis Berdasarkan Citra X-Ray Dual-View (AP + Lateral) dan Data Klinis Tabular Menggunakan Arsitektur Cross-Modal Gated Interaction**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.21](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Daftar Isi

- [Latar Belakang](#latar-belakang)
- [Arsitektur Model](#arsitektur-model)
- [Pipeline Preprocessing](#pipeline-preprocessing)
- [Dataset](#dataset)
- [Hasil Evaluasi](#hasil-evaluasi)
- [Struktur Proyek](#struktur-proyek)
- [Instalasi](#instalasi)
- [Menjalankan Aplikasi](#menjalankan-aplikasi)
- [File Model yang Dibutuhkan](#file-model-yang-dibutuhkan)
- [Interpretibilitas (XAI)](#interpretibilitas-xai)
- [Referensi Teknis](#referensi-teknis)

---

## Latar Belakang

Osteoporosis adalah penyakit degeneratif tulang yang ditandai dengan penurunan kepadatan massa tulang (*bone mineral density*) dan kerusakan mikroarsitektur jaringan tulang. Sering dijuluki sebagai **"silent disease"** karena berkembang tanpa gejala klinis yang nyata hingga pasien mengalami fraktur patologis pada vertebra lumbal, panggul, atau pergelangan tangan.

**Standar emas** diagnosis osteoporosis menurut WHO adalah pemeriksaan *Dual-energy X-ray Absorptiometry* (DXA) yang menghasilkan nilai *T-score*. Namun, ketersediaan mesin DXA sangat terbatas — tersentralisasi di rumah sakit rujukan tipe A, biayanya mahal, dan membutuhkan radiografer tersertifikasi khusus. Hal ini menyebabkan rendahnya cakupan deteksi dini di fasilitas pelayanan kesehatan primer dan daerah terpencil.

**LUMOS** memanfaatkan modalitas yang jauh lebih mudah diakses:
- **Foto rontgen (X-Ray) konvensional** vertebra lumbal dalam dua proyeksi: **AP (Anteroposterior)** dan **Lateral** — tersedia di hampir seluruh fasilitas kesehatan.
- **Data klinis rutin** (usia, jenis kelamin, tinggi badan, berat badan, BMI) — diperoleh dari anamnesis standar tanpa alat khusus.

Arsitektur **Cross-Modal Gated Interaction** dirancang untuk mengatasi kelemahan fusi konvensional (*simple concatenation*) di mana satu modalitas dapat mendominasi modalitas lain (*modality suppression*). Mekanisme gating sigmoid secara adaptif mengatur kontribusi representasi citra dan data klinis sebelum klasifikasi akhir.

---

## Arsitektur Model

### Varian D — Cross-Modal Gated Interaction (Model Terbaik)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        LUMOS — Arsitektur Multimodal (Varian D)                 │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  [AP X-Ray 224×224×3] → Rescale(255) → EfficientNetB0 → GAP → Dense(256)  ─┐   │
│                                                                              │   │
│                                                   Concat(512) → Dense(512 → │   │
│                                                   256 → 128 → 64) = h_img   │   │
│  [LAT X-Ray 224×224×3] → Rescale(255) → EfficientNetB0 → GAP → Dense(256) ─┘   │
│                                                                    │ (64-d)      │
│                                                                    │             │
│  [Data Klinis 5 fitur] → StandardScaler/OHE → Dense(64) →         │             │
│                           Dense(32) = h_tab ──────────────────────┐│             │
│                                                      (32-d)      ││             │
│                                                                   ▼▼             │
│                                        joint = [h_img ‖ h_tab] ∈ R⁹⁶            │
│                                              │              │                    │
│                                              ▼              ▼                    │
│                                  gate_img = σ(W·joint)  gate_tab = σ(W·joint)    │
│                                      ∈ [0,1]⁶⁴             ∈ [0,1]³²            │
│                                              │              │                    │
│                                              ▼              ▼                    │
│                                   h_img ⊙ gate_img    h_tab ⊙ gate_tab          │
│                                              │              │                    │
│                                              └──────┬───────┘                    │
│                                                     ▼                            │
│                                      fusion ∈ R⁹⁶ → Dense(128) → Dense(64)      │
│                                                     → Dense(3, Softmax)          │
│                                                     ▼                            │
│                                      [Normal │ Osteopenia │ Osteoporosis]        │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Studi Ablasi — 4 Varian Model

| Varian | Modalitas | Gating | Deskripsi |
|--------|-----------|--------|-----------|
| **A** · Clinical Only | Tabular | ❌ | MLP pada 5 fitur klinis (6 setelah OHE) |
| **B** · Image Only | AP + LAT | ❌ | Dual-stream EfficientNetB0 |
| **C** · Multimodal No-Gating | AP + LAT + Tab | ❌ | Fusi sederhana (*simple concat*) |
| **D** · Multimodal Gating ★ | AP + LAT + Tab | ✅ | **Cross-modal gated interaction** |

### Strategi Training — CheXNet-Style 2-Stage Fine-Tuning

| Stage | Epochs | Learning Rate | Backbone |
|-------|--------|---------------|----------|
| **Stage 1** — Linear Probe | 35 | 1×10⁻³ | Frozen (semua layer dibekukan) |
| **Stage 2** — Domain Adaptation | 35 | 1×10⁻⁵ (↓100×) | 30 layer teratas di-unfreeze |

- **Batch size**: 16
- **Loss**: `categorical_crossentropy` dengan *balanced class weights*
- **Optimizer**: Adam
- **Callbacks**: Custom `MacroOsteopeniaF1Checkpoint` (0.60×Macro-F1 + 0.40×F1-osteopenia), `EarlyStopping` (patience=7), `ReduceLROnPlateau` (factor=0.3, patience=4)

---

## Pipeline Preprocessing

### Preprocessing Citra X-Ray

```
Input .npy → Gaussian Blur → ROI Crop (L1–L4) → CLAHE + Otsu Bone Mask → Resize 224×224 → Normalize [0,1] → Stack RGB
```

| # | Langkah | Parameter | Fungsi |
|---|---------|-----------|--------|
| 1 | **Gaussian Filtering** | Kernel (5,5), σ=1.0 | Mereduksi *high-frequency noise* sensor DXA/radiografi |
| 2 | **ROI Cropping** | AP: top=12%, bottom=98%, left=22%, right=78% | Mengisolasi vertebra lumbal L1–L4 |
| | | LAT: top=12%, bottom=98%, left=12%, right=85% | |
| 3 | **CLAHE Enhancement** | clipLimit=2.5, tileGrid=(8,8) | Meningkatkan kontras struktur trabekular |
| 4 | **Otsu Bone Masking** | Threshold otomatis + morfologi (kernel 9×9) | Segmentasi tulang vs jaringan lunak |
| 5 | **Soft Blending** | mask_blend=0.65 | Meredam bayangan organ tanpa menghilangkan konteks |
| 6 | **Resize** | 224×224, interpolasi bilinear | Standarisasi dimensi input backbone |
| 7 | **Normalisasi** | Skala [0.0, 1.0] | Model mengharapkan input ternormalisasi |
| 8 | **Stack RGB** | Grayscale → 3 channel | Kompatibilitas dengan EfficientNetB0 pretrained ImageNet |

### Preprocessing Data Klinis Tabular

| Fitur | Tipe | Transformasi |
|-------|------|-------------|
| `age` (usia) | Numerik | StandardScaler |
| `height` (tinggi badan, m) | Numerik | StandardScaler |
| `weight` (berat badan, kg) | Numerik | StandardScaler |
| `bmi` (indeks massa tubuh) | Numerik | StandardScaler |
| `gender` (jenis kelamin) | Kategorik | OneHotEncoder → `gender_Female`, `gender_Male` |

> ⚠️ **Pencegahan Data Leakage**: Seluruh parameter hasil pengukuran DXA (BMD, BMC, T-score, Z-score, dsb.) **dieksklusi** dari fitur input karena merupakan variabel yang secara deterministik menentukan label target.

---

## Dataset

- **Sumber**: [damarsyarafiramadhan/lumos-data](https://www.kaggle.com/datasets/damarsyarafiramadhan/lumos-data)
- **Jumlah pasien valid**: **793** (setelah *index-safe merge* AP + Lateral + Tabular)
- **Format citra**: `.npy` (array NumPy)

### Distribusi Kelas

| Kelas | Jumlah | Persentase |
|-------|--------|------------|
| Normal | 289 | 36.4% |
| Osteopenia | 286 | 36.1% |
| Osteoporosis | 223 | 27.9% |

### Pembagian Data (Stratified Split)

| Set | Jumlah | Normal | Osteopenia | Osteoporosis |
|-----|--------|--------|------------|--------------|
| **Training** (80%) | 634 | 229 (36.1%) | 228 (36.0%) | 177 (27.9%) |
| **Validation** (15%) | 119 | 43 (36.1%) | 43 (36.1%) | 33 (27.7%) |
| **Test** (5%) | 40 | 15 (37.5%) | 14 (35.0%) | 11 (27.5%) |

### Index-Safe Pipeline

Seluruh quadruplet data `(AP_image, Lateral_image, tabular_row, label)` dikunci menggunakan `patient_id` unik untuk mencegah *index drift* atau *data mismatch* antar modalitas sepanjang split, shuffle, dan augmentasi.

---

## Hasil Evaluasi

Evaluasi pada *unseen test set* independen (40 sampel):

| Varian | Accuracy | Macro Precision | Macro Recall | Macro F1 | Macro AUC |
|--------|----------|-----------------|--------------|----------|-----------|
| A · Clinical Only | 62.50% | 65.92% | 63.95% | 57.84% | 85.55% |
| B · Image Only | 22.50% | 12.50% | 26.62% | 15.43% | 52.98% |
| C · Multimodal No-Gating | 62.50% | 63.29% | 63.29% | 62.49% | 70.96% |
| **D · Multimodal Gating** ★ | **67.50%** | **70.18%** | **68.70%** | **68.00%** | **86.03%** |

### Temuan Utama

- **Model D (Cross-Modal Gating)** unggul di semua metrik — *gating mechanism* berhasil menyeimbangkan kontribusi antar modalitas.
- Peningkatan F1-Score: **+10.16%** vs Clinical Only, **+5.51%** vs Multimodal No-Gating.
- Peningkatan AUC: **+15.07%** vs Multimodal No-Gating.
- Image Only (Model B) sangat rendah secara mandiri (F1 15.43%), tetapi berkontribusi signifikan sebagai fitur komplementer dalam fusi multimodal.

---

## Struktur Proyek

```
klasifikasi-multimodal-osteoporosis/
│
├── app.py                              # Aplikasi Streamlit utama (inferensi)
├── requirements.txt                    # Dependensi Python
├── README.md                           # Dokumentasi proyek (file ini)
├── lumos-bismillah-final-effnet.ipynb  # Notebook training & evaluasi
│
├── deploy_artifacts/                   # Artefak deployment
│   ├── best_D_Multimodal_Gating.keras  # Model terbaik (checkpoint deploy)
│   ├── tab_preprocessor.pkl            # ColumnTransformer (StandardScaler + OHE)
│   ├── metadata.json                   # Konfigurasi model & preprocessing
│   ├── preprocessing.py                # Modul preprocessing citra (standalone)
│   ├── ablation_results.csv            # Hasil studi ablasi
│   ├── shap_feature_importance.csv     # Ranking kontribusi fitur SHAP
│   ├── training_curves.png             # Kurva loss & akurasi training
│   └── *.png                           # Visualisasi evaluasi & Grad-CAM
│
├── model_lumas_dual/                   # Model lengkap (semua varian)
│   ├── best_A_Clinical_Only.keras      # Varian A: Clinical Only
│   ├── best_B_Image_Only.keras         # Varian B: Image Only
│   ├── best_C_Multimodal_NoGating.keras # Varian C: Multimodal tanpa gating
│   ├── best_D_Multimodal_Gating.keras  # Varian D: Multimodal + Gating
│   └── *.csv, *.png                    # Data evaluasi & visualisasi
│
└── .venv/                              # Virtual environment Python 3.12
```

---

## Instalasi

### Prasyarat

- **Python 3.12** (direkomendasikan via [uv](https://docs.astral.sh/uv/) atau [pyenv](https://github.com/pyenv/pyenv))
- **Git LFS** (opsional, untuk clone model besar dari repo)

### Langkah Instalasi

```bash
# 1. Clone repository
git clone https://github.com/dmareee/klasifikasi-multimodal-osteoporosis.git
cd klasifikasi-multimodal-osteoporosis

# 2. Buat virtual environment (menggunakan uv — direkomendasikan)
uv venv --python 3.12
# atau menggunakan venv bawaan Python:
# python -m venv .venv

# 3. Aktifkan virtual environment
# Windows (PowerShell):
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# 4. Install dependensi
pip install -r requirements.txt
```

### Versi Library Terverifikasi

| Library | Versi Minimum | Versi Terverifikasi |
|---------|---------------|---------------------|
| Python | 3.10+ | **3.12.13** |
| TensorFlow | ≥2.15 | **2.21.0** |
| Keras | (bundled TF) | **3.15.1** |
| Streamlit | ≥1.35 | 1.35+ |
| OpenCV | ≥4.9 | 4.9+ |
| NumPy | ≥1.26 | 1.26+ |
| Pandas | ≥2.2 | 2.2+ |
| scikit-learn | ≥1.4 | 1.4+ |

---

## Menjalankan Aplikasi

```bash
# Pastikan virtual environment aktif, lalu:
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`.

### Input yang Diperlukan

1. **Data Klinis**: Usia, tinggi badan (m), berat badan (kg), jenis kelamin — BMI dihitung otomatis.
2. **Citra X-Ray**: Upload 2 file `.npy` — satu untuk proyeksi **AP** dan satu untuk proyeksi **Lateral**.

### Output

- **Prediksi kelas**: Normal / Osteopenia / Osteoporosis dengan confidence score
- **Perbandingan model**: Confidence dari semua varian yang tersedia
- **Grad-CAM**: Heatmap atensi visual pada citra AP dan Lateral
- **Probabilitas per kelas**: Distribusi skor prediksi untuk ketiga kelas

---

## File Model yang Dibutuhkan

Minimal untuk menjalankan aplikasi:

| File | Wajib | Lokasi |
|------|-------|--------|
| `best_D_Multimodal_Gating.keras` | ✅ **Ya** | `deploy_artifacts/` atau `model_lumas_dual/` |
| `tab_preprocessor.pkl` | ✅ **Ya** | `deploy_artifacts/` |

Opsional (untuk perbandingan antar varian di UI):

| File | Lokasi |
|------|--------|
| `best_A_Clinical_Only.keras` | `model_lumas_dual/` |
| `best_B_Image_Only.keras` | `model_lumas_dual/` |
| `best_C_Multimodal_NoGating.keras` | `model_lumas_dual/` |

> **Catatan**: `app.py` memiliki mekanisme `_resolve_artifact()` yang secara otomatis mencari file model di beberapa lokasi: `deploy_artifacts/`, `model_lumas_dual/`, dan folder root proyek.

---

## Interpretibilitas (XAI)

### Dual-View Grad-CAM

Visualisasi *Gradient-weighted Class Activation Mapping* (Selvaraju et al., 2017) pada layer konvolusi terakhir masing-masing backbone EfficientNetB0:
- Menghasilkan heatmap atensi terpisah untuk proyeksi **AP** dan **Lateral**
- Overlay colormap JET dengan transparansi α=0.55 di atas citra asli
- Memverifikasi bahwa model fokus pada korpus vertebra lumbal yang relevan secara anatomis

### SHAP Analysis

Atribusi kontribusi fitur klinis tabular menggunakan `KernelExplainer`:

| Ranking | Fitur | Keterangan |
|---------|-------|------------|
| 1 | `age` | Fitur paling berpengaruh |
| 2 | `bmi` | Korelasi kuat dengan kepadatan tulang |
| 3 | `weight` | Berat badan sebagai faktor protektif |
| 4 | `height` | Tinggi badan |
| 5 | `gender_Female` | Wanita memiliki risiko lebih tinggi |
| 6 | `gender_Male` | — |

---

## Augmentasi Data

Diterapkan **hanya** pada set training:

| Teknik | Parameter |
|--------|-----------|
| Random Rotation | ±25° |
| Random Width Shift | ±10% |
| Random Height Shift | ±10% |
| Random Shear | ±0.12 |
| Horizontal Flip | 50% probabilitas |
| Vertical Flip | ❌ Tidak digunakan |

---

## Referensi Teknis

- **Backbone**: [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946) (Tan & Le, 2019)
- **Transfer Learning**: [CheXNet: Radiologist-Level Pneumonia Detection](https://arxiv.org/abs/1711.05225) (Rajpurkar et al., 2017)
- **Grad-CAM**: [Gradient-weighted Class Activation Mapping](https://arxiv.org/abs/1610.02391) (Selvaraju et al., 2017)
- **SHAP**: [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874) (Lundberg & Lee, 2017)
- **Dataset**: [LUMOS Data — Kaggle](https://www.kaggle.com/datasets/damarsyarafiramadhan/lumos-data)

---

## Lisensi

Proyek ini dikembangkan sebagai bagian dari penelitian skripsi. Silakan hubungi penulis untuk penggunaan di luar konteks akademis.

---

<p align="center">
  <b>LUMOS</b> — Lumbar Spine Osteoporosis Screening System<br>
  <i>Klasifikasi Multimodal Osteoporosis berbasis Deep Learning</i>
</p>