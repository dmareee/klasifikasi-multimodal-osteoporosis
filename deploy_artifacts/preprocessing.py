"""
LUMOS — Preprocessing Module
=============================
Pipeline preprocessing citra X-Ray dan data klinis tabular untuk inferensi
model klasifikasi osteoporosis multimodal.

Modul ini mengimplementasikan pipeline identik dengan training di notebook
`lumos-bismillah-final-effnet.ipynb`, terdiri dari:

  Citra:   Gaussian Filter → ROI Crop (L1–L4) → CLAHE + Otsu Bone Mask
           → Resize 224×224 → Normalize [0,1] → Stack RGB

  Tabular: StandardScaler (numerik) + OneHotEncoder (kategorik) via
           tab_preprocessor.pkl

Dipakai oleh: notebook (source of truth) & app.py (Streamlit inference).
"""

import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Konstanta — identik dengan notebook (cell "Configuration & Constants")
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)           # (H, W)
GAUSSIAN_KERNEL_SIZE = (5, 5)
GAUSSIAN_SIGMA = 1.0

ROI_CONFIG = {
    "ap":  {"top": 0.12, "bottom": 0.98, "left": 0.22, "right": 0.78},
    "lat": {"top": 0.12, "bottom": 0.98, "left": 0.12, "right": 0.85},
}

CLASS_NAMES = ["normal", "osteopenia", "osteoporosis"]
LABEL_MAP = {"normal": 0, "osteopenia": 1, "osteoporosis": 2}
N_CLASSES = 3

# Backbone EfficientNetB0 mengharapkan input [0, 255]; layer Rescaling(255)
# di dalam model akan mengalikan input [0, 1] menjadi [0, 255] sebelum masuk
# ke normalisasi bawaan EfficientNetB0.
BACKBONE_INPUT_SCALE = 255.0


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pemuatan Citra — load_image_array
# ─────────────────────────────────────────────────────────────────────────────
def load_image_array(path, mode="npy"):
    """
    Membaca file citra X-Ray menjadi array 2D grayscale float32.

    Parameters
    ----------
    path : str atau Path
        Jalur file citra (.npy atau .dcm).
    mode : {'npy', 'dcm'}, default 'npy'
        Format file input.

    Returns
    -------
    np.ndarray
        Array 2D (H, W) float32 grayscale.
    """
    path = Path(path)

    if mode == "dcm":
        import pydicom
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
    else:
        arr = np.load(str(path)).astype(np.float32)

    # Jika file 3D (volume/multi-slice), ambil irisan tengah
    if arr.ndim == 3 and arr.shape[0] not in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = arr[arr.shape[0] // 2]

    # Pastikan 2D grayscale
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]

    return arr.astype(np.float32)


def load_image_array_from_bytes(file_bytes, filename="upload.png"):
    """
    Membaca citra dari bytes (upload PNG/JPG via Streamlit) menjadi
    array 2D grayscale float32.

    Parameters
    ----------
    file_bytes : bytes
        Raw bytes dari file citra yang di-upload.
    filename : str
        Nama file untuk menentukan metode decode.

    Returns
    -------
    np.ndarray
        Array 2D (H, W) float32 grayscale.
    """
    buf = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Gagal decode citra dari bytes: {filename}")
    return img.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Gaussian Filtering — Pengurangan Derau Sensor
# ─────────────────────────────────────────────────────────────────────────────
def apply_gaussian_filter(arr, ksize=GAUSSIAN_KERNEL_SIZE, sigma=GAUSSIAN_SIGMA):
    """
    Menerapkan Gaussian Blur untuk mereduksi high-frequency noise sensor
    radiografi/DXA.

    Citra dinormalisasi ke uint8 [0, 255] terlebih dahulu, lalu di-blur
    dengan kernel (5, 5) dan σ=1.0.

    Parameters
    ----------
    arr : np.ndarray
        Array 2D grayscale float32.
    ksize : tuple of int, default (5, 5)
        Ukuran kernel Gaussian.
    sigma : float, default 1.0
        Standar deviasi Gaussian.

    Returns
    -------
    np.ndarray
        Array 2D float32 setelah Gaussian filtering.
    """
    arr_norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.GaussianBlur(arr_norm, ksize, sigma).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ROI Cropping — Pemotongan Area Vertebra Lumbal L1–L4
# ─────────────────────────────────────────────────────────────────────────────
def crop_lumbar_roi(arr, view="ap"):
    """
    Memotong area vertebra lumbal L1–L4 berdasarkan koordinat persentase
    yang telah ditentukan dalam ROI_CONFIG.

    Koordinat ROI:
      AP  → Top 12%, Bottom 98%, Left 22%, Right 78%
      LAT → Top 12%, Bottom 98%, Left 12%, Right 85%

    Parameters
    ----------
    arr : np.ndarray
        Array 2D citra setelah Gaussian filtering.
    view : {'ap', 'lat'}, default 'ap'
        Proyeksi citra X-Ray.

    Returns
    -------
    np.ndarray
        Array 2D hasil cropping area lumbal.
    """
    cfg = ROI_CONFIG.get(view, ROI_CONFIG["ap"])
    h, w = arr.shape[:2]
    r0, r1 = int(h * cfg["top"]), int(h * cfg["bottom"])
    c0, c1 = int(w * cfg["left"]), int(w * cfg["right"])
    cropped = arr[r0:r1, c0:c1]
    return cropped if cropped.size > 0 else arr


# ─────────────────────────────────────────────────────────────────────────────
# 4. Bone Enhancement — CLAHE + Otsu Masking + Morfologi
# ─────────────────────────────────────────────────────────────────────────────
def enhance_bone_region(arr, clip_limit=2.5, tile_size=(8, 8), mask_blend=0.65):
    """
    Meningkatkan kontras struktur trabekular tulang menggunakan:
      1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
      2. Otsu Thresholding — segmentasi otomatis area tulang berdensitas tinggi
      3. Operasi morfologi — Morphological Closing (menutup rongga trabekular)
         dan Opening (menghilangkan artefak kecil)
      4. Soft blending — menggabungkan citra CLAHE dengan bone mask

    Parameters
    ----------
    arr : np.ndarray
        Array 2D citra setelah ROI cropping.
    clip_limit : float, default 2.5
        Batas clipping CLAHE.
    tile_size : tuple of int, default (8, 8)
        Ukuran grid tile CLAHE.
    mask_blend : float, default 0.65
        Faktor blending antara bone mask dan citra enhanced.
        Formula: enhanced × (mask × 0.65 + 0.35)

    Returns
    -------
    np.ndarray
        Array 2D float32 dengan kontras tulang yang ditingkatkan.
    """
    # Normalisasi ke uint8 jika diperlukan
    if arr.dtype != np.uint8:
        arr_u8 = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        arr_u8 = arr.copy()

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    enhanced = clahe.apply(arr_u8)

    # Otsu Thresholding — segmentasi area tulang keras
    _, bone_mask = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Operasi Morfologi
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bone_mask = cv2.morphologyEx(bone_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    bone_mask = cv2.morphologyEx(bone_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Soft blending — meredam jaringan lunak tanpa menghilangkan konteks
    mask_norm = bone_mask.astype(np.float32) / 255.0
    soft_blend = mask_norm * mask_blend + (1.0 - mask_blend)
    return enhanced.astype(np.float32) * soft_blend


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pipeline Lengkap — preprocess_image
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_image(scan_array, view="ap", img_size=IMG_SIZE):
    """
    Pipeline preprocessing citra X-Ray lengkap:
        Gaussian Filter → ROI Crop → CLAHE/Otsu Enhancement
        → Resize → Normalize [0, 1] → Stack RGB → Expand batch dim

    CATATAN PENTING soal skala input model:
    Di dalam model, layer `rescale_ap`/`rescale_lat` mengalikan input
    dengan 255, lalu backbone EfficientNetB0 punya Rescaling(1/255)
    bawaan — keduanya saling meniadakan. Artinya model mengharapkan
    gambar yang sudah dinormalisasi ke rentang [0, 1], seperti yang
    dihasilkan oleh fungsi ini.

    Parameters
    ----------
    scan_array : np.ndarray
        Array 2D/3D citra X-Ray (raw dari .npy atau .dcm).
    view : {'ap', 'lat'}, default 'ap'
        Proyeksi citra (menentukan koordinat ROI).
    img_size : tuple of int, default (224, 224)
        Ukuran target output (H, W).

    Returns
    -------
    np.ndarray
        Array float32 dengan shape (1, H, W, 3) siap untuk inferensi model.
    """
    arr = np.asarray(scan_array, dtype=np.float32)

    # Pastikan input 2D grayscale
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    elif arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(
            np.asarray(arr, dtype=np.uint8), cv2.COLOR_BGR2GRAY
        ).astype(np.float32)

    # Pipeline preprocessing
    arr = apply_gaussian_filter(arr)
    arr = crop_lumbar_roi(arr, view=view)
    arr = enhance_bone_region(arr)

    # Resize ke dimensi target
    arr = cv2.resize(
        arr, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR
    )
    arr = arr.astype(np.float32)

    # Normalisasi ke [0, 1]
    if arr.max() > 1.0:
        arr = arr / 255.0

    # Stack ke 3 channel (RGB) — kompatibel dengan EfficientNetB0 pretrained
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    # Expand dimensi batch: (H, W, 3) → (1, H, W, 3)
    arr = np.expand_dims(arr, axis=0)

    return arr.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Preprocessing Tabular — prepare_tabular
# ─────────────────────────────────────────────────────────────────────────────
def load_tab_preprocessor(pkl_path):
    """
    Memuat ColumnTransformer (StandardScaler + OneHotEncoder) dari file pickle
    yang dihasilkan saat training.

    Menangani kompatibilitas antar versi scikit-learn di mana atribut
    internal `_RemainderColsList` mungkin tidak tersedia.

    Parameters
    ----------
    pkl_path : str atau Path
        Jalur file tab_preprocessor.pkl.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Preprocessor tabular yang sudah di-fit pada data training.
    """
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"tab_preprocessor.pkl tidak ditemukan: {pkl_path}")

    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        # Kompatibilitas versi sklearn yang lebih baru
        if "_RemainderColsList" not in str(exc):
            raise
        import sklearn.compose._column_transformer as ct_module

        if not hasattr(ct_module, "_RemainderColsList"):

            class _CompatRemainderColsList(list):
                pass

            ct_module._RemainderColsList = _CompatRemainderColsList

        with open(pkl_path, "rb") as f:
            return pickle.load(f)


def prepare_tabular(age, height, weight, bmi, gender, preprocessor=None):
    """
    Memproses data klinis pasien menjadi array fitur siap inferensi.

    PENTING: OneHotEncoder di-fit dengan kategori 'Female'/'Male'
    (title-case), bukan 'female'/'male'. Gender dinormalisasi ke
    title-case sebelum transformasi.

    Fitur input (5 raw → 6 output setelah OHE):
      Numerik:    age, height, weight, bmi    → StandardScaler
      Kategorik:  gender                      → OneHotEncoder → gender_Female, gender_Male

    Parameters
    ----------
    age : float
        Usia pasien (tahun).
    height : float
        Tinggi badan (meter).
    weight : float
        Berat badan (kilogram).
    bmi : float
        Indeks Massa Tubuh (kg/m²).
    gender : str
        Jenis kelamin ('female' atau 'male').
    preprocessor : sklearn.compose.ColumnTransformer, optional
        Preprocessor yang sudah di-fit. Jika None, harus di-load terlebih
        dahulu menggunakan load_tab_preprocessor().

    Returns
    -------
    dict
        - 'full': np.ndarray float32 shape (1, 6) — fitur siap inferensi
        - 'raw': pd.DataFrame — data mentah sebelum transformasi
    """
    if preprocessor is None:
        raise RuntimeError(
            "tab_preprocessor.pkl tidak ditemukan. File ini wajib ada supaya "
            "fitur tabular diskalakan (StandardScaler) dan di-encode "
            "(OneHotEncoder) persis seperti saat training."
        )

    # Normalisasi gender ke title-case sesuai fit OHE
    gender_title = str(gender).strip().capitalize()  # "female" → "Female"

    feature_df = pd.DataFrame(
        [
            {
                "age": float(age),
                "height": float(height),
                "weight": float(weight),
                "bmi": float(bmi),
                "gender": gender_title,
            }
        ]
    )

    transformed = preprocessor.transform(feature_df)

    return {
        "full": np.asarray(transformed, dtype=np.float32),
        "raw": feature_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Utilitas tambahan
# ─────────────────────────────────────────────────────────────────────────────
def read_npy_scan(uploaded_file):
    """
    Membaca file .npy yang di-upload via Streamlit.

    Parameters
    ----------
    uploaded_file : streamlit.UploadedFile
        File .npy yang di-upload pengguna.

    Returns
    -------
    np.ndarray
        Array float32 citra X-Ray.
    """
    uploaded_file.seek(0)
    arr = np.load(uploaded_file)

    # Volume 3D (multi-slice) → ambil irisan tengah
    if arr.ndim == 3 and arr.shape[0] not in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = arr[arr.shape[0] // 2]

    return arr.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Eksekusi langsung — demonstrasi pipeline
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("LUMOS Preprocessing Module")
    print("=" * 50)
    print(f"  IMG_SIZE            : {IMG_SIZE}")
    print(f"  GAUSSIAN_KERNEL     : {GAUSSIAN_KERNEL_SIZE}")
    print(f"  GAUSSIAN_SIGMA      : {GAUSSIAN_SIGMA}")
    print(f"  ROI_CONFIG (AP)     : {ROI_CONFIG['ap']}")
    print(f"  ROI_CONFIG (LAT)    : {ROI_CONFIG['lat']}")
    print(f"  CLASS_NAMES         : {CLASS_NAMES}")
    print(f"  BACKBONE_INPUT_SCALE: {BACKBONE_INPUT_SCALE}")
    print()
    print("Pipeline citra:")
    print("  1. load_image_array()      - Baca .npy/.dcm -> float32 2D")
    print("  2. apply_gaussian_filter() - Gaussian Blur (5x5, sigma=1.0)")
    print("  3. crop_lumbar_roi()       - ROI crop L1-L4")
    print("  4. enhance_bone_region()   - CLAHE + Otsu + Morfologi")
    print("  5. preprocess_image()      - Resize + Normalize + RGB + Batch")
    print()
    print("Pipeline tabular:")
    print("  1. load_tab_preprocessor() - Load ColumnTransformer .pkl")
    print("  2. prepare_tabular()       - Scale + OHE -> array (1, 6)")
