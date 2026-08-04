"""
LUMOS — Preprocessing Module
Persis sama dengan pipeline training di notebook (cell ROI Configuration / Gaussian / Bone Enhancement).
Dipakai bareng oleh: notebook (source of truth) & streamlit_app.py (inference).
"""

import cv2
import numpy as np
import pydicom


# ══════════════════════════════════════════════════════════════════════════
#  CONFIG (harus sama persis dengan notebook cell config)
# ══════════════════════════════════════════════════════════════════════════
IMG_H, IMG_W = 224, 224
IMG_SIZE = (IMG_H, IMG_W)

GAUSSIAN_KERNEL_SIZE = (5, 5)
GAUSSIAN_SIGMA = 1.0

ROI_CONFIG = {
    "ap":  {"top": 0.12, "bottom": 0.98, "left": 0.22, "right": 0.78},
    "lat": {"top": 0.12, "bottom": 0.98, "left": 0.12, "right": 0.85},
}

NUM_FEATURES = ["age", "height", "weight", "bmi"]
CATEGORICAL_FEATURES = ["gender"]

CLASS_NAMES = ["normal", "osteopenia", "osteoporosis"]
LABEL_MAP = {"normal": 0, "osteopenia": 1, "osteoporosis": 2}


# ══════════════════════════════════════════════════════════════════════════
#  IMAGE LOADING
# ══════════════════════════════════════════════════════════════════════════
def load_image_array(path: str, mode: str = "npy") -> np.ndarray:
    """Load gambar dari .npy atau .dcm dan kembalikan array 2D grayscale."""
    if mode == "npy":
        arr = np.load(path)
        if arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]
    else:
        ds = pydicom.dcmread(path, force=True)
        arr = ds.pixel_array
    return arr.astype(np.float32)


def load_image_array_from_bytes(file_bytes: bytes) -> np.ndarray:
    """Load gambar dari file upload biasa (png/jpg) untuk keperluan deployment."""
    buf = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    arr = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return arr.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════
#  PIPELINE STEPS — identik dengan notebook
# ══════════════════════════════════════════════════════════════════════════
def apply_gaussian_filter(arr, ksize=GAUSSIAN_KERNEL_SIZE, sigma=GAUSSIAN_SIGMA):
    """Gaussian Blur untuk noise reduction sensor DXA."""
    arr_norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.GaussianBlur(arr_norm, ksize, sigma).astype(np.float32)


def crop_lumbar_roi(arr, view="ap"):
    """Crop ROI vertebra lumbal L1-L4 (koordinat persentase)."""
    cfg = ROI_CONFIG.get(view, ROI_CONFIG["ap"])
    h, w = arr.shape[:2]
    r0, r1 = int(h * cfg["top"]), int(h * cfg["bottom"])
    c0, c1 = int(w * cfg["left"]), int(w * cfg["right"])
    cropped = arr[r0:r1, c0:c1]
    return cropped if cropped.size > 0 else arr


def enhance_bone_region(arr, clip_limit=2.5, tile_size=(8, 8), mask_blend=0.65):
    """CLAHE + Otsu masking: tingkatkan kontras tulang, lemahkan soft tissue."""
    if arr.dtype != np.uint8:
        arr_u8 = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        arr_u8 = arr.copy()
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    enhanced = clahe.apply(arr_u8)
    _, bone_mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bone_mask = cv2.morphologyEx(bone_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    bone_mask = cv2.morphologyEx(bone_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_norm = bone_mask.astype(np.float32) / 255.0
    soft_blend = mask_norm * mask_blend + (1.0 - mask_blend)
    return enhanced.astype(np.float32) * soft_blend


def preprocess_image(path, mode="npy", target_size=IMG_SIZE,
                      view="ap", apply_filter=True,
                      apply_roi=True, apply_bone=True):
    """Pipeline lengkap (dari path file, dipakai saat training):
    Load -> Gaussian -> ROI Crop -> Bone Enhancement -> Resize -> Normalize -> RGB
    """
    arr = load_image_array(path, mode)
    return _run_pipeline(arr, target_size, view, apply_filter, apply_roi, apply_bone)


def preprocess_uploaded_image(file_bytes, target_size=IMG_SIZE, view="ap",
                               apply_filter=True, apply_roi=True, apply_bone=True):
    """Pipeline lengkap dari bytes upload (dipakai saat inference/Streamlit)."""
    arr = load_image_array_from_bytes(file_bytes)
    return _run_pipeline(arr, target_size, view, apply_filter, apply_roi, apply_bone)


def _run_pipeline(arr, target_size, view, apply_filter, apply_roi, apply_bone):
    if apply_filter:
        arr = apply_gaussian_filter(arr)
    if apply_roi:
        arr = crop_lumbar_roi(arr, view=view)
    if apply_bone:
        arr = enhance_bone_region(arr)
    arr_resized = cv2.resize(arr, target_size, interpolation=cv2.INTER_LINEAR)
    arr_norm = arr_resized / 255.0 if arr_resized.max() > 1.0 else arr_resized
    arr_norm = arr_norm.astype(np.float32)
    if arr_norm.ndim == 2:
        arr_rgb = np.stack([arr_norm] * 3, axis=-1)
    else:
        arr_rgb = arr_norm
    return arr_rgb


# ══════════════════════════════════════════════════════════════════════════
#  TABULAR HELPER
# ══════════════════════════════════════════════════════════════════════════
def compute_bmi(weight_kg: float, height_cm: float) -> float:
    return round(weight_kg / ((height_cm / 100) ** 2), 2)
