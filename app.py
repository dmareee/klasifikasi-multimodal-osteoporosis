import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import keras

st.set_page_config(
    page_title="Osteoporosis Multi-Modal Detection",
    page_icon="🦴",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────
# Paths — semua artefak (model, tab_preprocessor.pkl) ada SEJAJAR dengan
# app.py ini, bukan di subfolder "model_lumas_dual" / "deploy_artifacts".
# Tidak ada metadata.json yang dikirim, jadi konstanta di-hardcode di sini
# persis sama dengan notebook (cell "Configuration & Constants" dan modul
# preprocessing.py yang diekspor notebook).
# ─────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent


def _resolve_artifact(filename):
    """
    Cari file artefak dengan urutan prioritas:
    1) path RELATIF terhadap current working directory — ini yang dipakai
       versi app.py sebelumnya (mis. os.path.join("deploy_artifacts", ...))
       dan biasanya cocok kalau `streamlit run` dijalankan dari root
       project (E:\\...\\klasifikasi-multimodal-osteoporosis\\).
    2) relatif terhadap cwd tanpa subfolder deploy_artifacts.
    3) relatif terhadap folder app.py (Path(__file__).parent) — fallback
       untuk kasus cwd beda dari lokasi app.py.
    4) pencarian rekursif di bawah cwd dan BASE_DIR.
    """
    candidates = [
        os.path.join("deploy_artifacts", filename),
        os.path.join("model_lumas_dual", filename),
        filename,
        BASE_DIR / "deploy_artifacts" / filename,
        BASE_DIR / "model_lumas_dual" / filename,
        BASE_DIR / filename,
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            return p

    for root in (Path.cwd(), BASE_DIR):
        try:
            for found in root.rglob(filename):
                return found
        except Exception:
            pass

    # Tidak ketemu di mana pun — kembalikan kandidat pertama supaya pesan
    # error di UI tetap menunjukkan path yang jelas.
    return Path(candidates[0])


def _debug_dir_listing(path, max_items=30):
    """Bantu diagnosa: tampilkan isi folder apa adanya (untuk sidebar)."""
    try:
        if not path.exists():
            return f"(folder tidak ada: {path})"
        items = sorted(os.listdir(path))[:max_items]
        return "\n".join(items) if items else "(folder kosong)"
    except Exception as exc:
        return f"(gagal membaca folder: {exc})"


def _artifact_cache_key(path):
    """Buat cache key yang berubah saat file benar-benar muncul atau berubah."""
    p = Path(path)
    try:
        if p.exists():
            return str(p.resolve()), p.stat().st_mtime_ns
    except Exception:
        pass
    return str(p), -1


MODEL_PATH = _resolve_artifact("best_D_Multimodal_Gating.keras")
TAB_PREPROCESSOR_PATH = _resolve_artifact("tab_preprocessor.pkl")

IMG_SIZE = (224, 224)  # (H, W)
GAUSSIAN_KERNEL_SIZE = (5, 5)
GAUSSIAN_SIGMA = 1.0
# ── ROI Configuration ────────────────────────────────────────────────────
ROI_CONFIG = {
    'ap':  {'top': 0.12, 'bottom': 1, 'left': 0.22, 'right': 0.78},
    'lat': {'top': 0.12, 'bottom': 1, 'left': 0.12, 'right': 0.85},
}

CLASS_NAMES = ["normal", "osteopenia", "osteoporosis"]
LABEL_LABELS = {
    "normal": "Normal",
    "osteopenia": "Osteopenia",
    "osteoporosis": "Osteoporosis",
}

# Nama layer & input SEBENARNYA di best_D_Multimodal_Gating.keras (dicek
# langsung dari config.json model — arsitekturnya EfficientNetB0, BUKAN
# ResNet50). Model ini tidak punya cabang clinical-only / image-only
# terpisah: satu-satunya checkpoint yang dikirim adalah model gabungan D,
# jadi aplikasi ini SELALU butuh AP + Lateral + data klinis sekaligus.
INPUT_AP_NAME = "input_ap"
INPUT_LAT_NAME = "input_lat"
INPUT_TAB_NAME = "input_tab"
BRANCH_AP_NAME = "efficientnetb0_ap"
BRANCH_LAT_NAME = "efficientnetb0_lat"


# ─────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str, mtime_ns: int):
    path = Path(model_path)
    if not path.exists():
        return None
    return tf.keras.models.load_model(str(path), compile=False)


@st.cache_resource
def build_gradcam_model(_model):
    """
    Bangun ulang forward pass model D SECARA MANUAL, layer demi layer,
    dari Input baru, supaya conv_ap / conv_lat (output backbone
    EfficientNetB0 sebelum GAP) berada dalam SATU graph yang sama dengan
    output akhir "output". Ini perlu dilakukan begini (bukan sekadar
    `Model(inputs=model.inputs, outputs=[branch.output, model.output])`)
    karena setelah model .keras di-load ulang, tensor `branch.output` tidak
    lagi terhubung ke graph baru di Keras 3 — sehingga gradient-nya None.
    Urutan layer di bawah mengikuti persis inbound_nodes pada config.json
    model (lihat arsitektur: dual EfficientNetB0 -> gating fusion -> output).
    """
    if _model is None:
        return None

    g = _model.get_layer

    inp_ap = tf.keras.Input(shape=(224, 224, 3), name="ga_input_ap")
    inp_lat = tf.keras.Input(shape=(224, 224, 3), name="ga_input_lat")
    inp_tab = tf.keras.Input(shape=(6,), name="ga_input_tab")

    x_ap = g("rescale_ap")(inp_ap)
    conv_ap = g(BRANCH_AP_NAME)(x_ap)  # target Grad-CAM AP
    x_lat = g("rescale_lat")(inp_lat)
    conv_lat = g(BRANCH_LAT_NAME)(x_lat)  # target Grad-CAM Lateral

    gap_ap = g("gap_ap")(conv_ap)
    gap_lat = g("gap_lat")(conv_lat)
    bn_ap = g("bn_ap")(gap_ap, training=False)
    bn_lat = g("bn_lat")(gap_lat, training=False)
    dense_ap = g("dense_ap")(bn_ap)
    dense_lat = g("dense_lat")(bn_lat)
    drop_ap = g("drop_ap")(dense_ap, training=False)
    drop_lat = g("drop_lat")(dense_lat, training=False)
    img_concat = g("img_concat")([drop_ap, drop_lat])

    img_d512 = g("img_d512")(img_concat)
    img_bn512 = g("img_bn512")(img_d512, training=False)
    img_drop512 = g("img_drop512")(img_bn512, training=False)
    img_d256 = g("img_d256")(img_drop512)
    img_bn256 = g("img_bn256")(img_d256, training=False)
    img_drop256 = g("img_drop256")(img_bn256, training=False)
    img_d128 = g("img_d128")(img_drop256)
    img_bn128 = g("img_bn128")(img_d128, training=False)
    img_drop128 = g("img_drop128")(img_bn128, training=False)
    img_d64 = g("img_d64")(img_drop128)
    img_bn64 = g("img_bn64")(img_d64, training=False)
    img_drop64 = g("img_drop64")(img_bn64, training=False)
    h_img = g("h_img")(img_drop64)

    tab_d1 = g("tab_d1")(inp_tab)
    tab_bn1 = g("tab_bn1")(tab_d1, training=False)
    dropout = g("dropout")(tab_bn1, training=False)
    tab_d2 = g("tab_d2")(dropout)
    tab_bn2 = g("tab_bn2")(tab_d2, training=False)
    tab_drop1 = g("tab_drop1")(tab_bn2, training=False)

    joint_concat = g("joint_concat")([h_img, tab_drop1])
    gate_img = g("gate_img")(joint_concat)
    gate_tab = g("gate_tab")(joint_concat)
    img_gated = g("img_gated")([h_img, gate_img])
    tab_gated = g("tab_gated")([tab_drop1, gate_tab])
    cross_gated_fusion = g("cross_gated_fusion")([img_gated, tab_gated])

    fusion_d64128 = g("fusion_d64128")(cross_gated_fusion)
    fusion_bn64128 = g("fusion_bn64128")(fusion_d64128, training=False)
    fusion_drop64128 = g("fusion_drop64128")(fusion_bn64128, training=False)
    fusion_d64 = g("fusion_d64")(fusion_drop64128)
    fusion_bn64 = g("fusion_bn64")(fusion_d64, training=False)
    fusion_drop64 = g("fusion_drop64")(fusion_bn64, training=False)
    output = g("output")(fusion_drop64)

    return tf.keras.Model(
        inputs=[inp_ap, inp_lat, inp_tab],
        outputs=[conv_ap, conv_lat, output],
    )


@st.cache_resource
def load_tab_preprocessor(model_path: str, mtime_ns: int):
    path = Path(model_path)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        # Kompatibilitas versi sklearn yang lebih baru — atribut internal
        # ColumnTransformer berubah nama.
        if "_RemainderColsList" not in str(exc):
            raise
        import sklearn.compose._column_transformer as ct_module

        if not hasattr(ct_module, "_RemainderColsList"):
            class _CompatRemainderColsList(list):
                pass

            ct_module._RemainderColsList = _CompatRemainderColsList
        with open(path, "rb") as f:
            return pickle.load(f)


MODEL = None
TAB_PREPROCESSOR = None
GRADCAM_MODEL = None
MODEL_LOAD_ERROR = None
GRADCAM_ERROR = None
MODEL_VARIANTS = {}

try:
    MODEL = load_model(*_artifact_cache_key(MODEL_PATH))
except Exception as exc:
    MODEL_LOAD_ERROR = f"Gagal memuat model utama: {exc}"

try:
    TAB_PREPROCESSOR = load_tab_preprocessor(*_artifact_cache_key(TAB_PREPROCESSOR_PATH))
except Exception as exc:
    TAB_PREPROCESSOR = None

for variant_name, artifact_name in {
    "Clinical Only": "best_A_Clinical_Only.keras",
    "Image Only": "best_B_Image_Only.keras",
    "Multimodal No Gating": "best_C_Multimodal_NoGating.keras",
    "Multimodal Gating": "best_D_Multimodal_Gating.keras",
}.items():
    path = _resolve_artifact(artifact_name)
    try:
        MODEL_VARIANTS[variant_name] = load_model(*_artifact_cache_key(path))
    except Exception:
        MODEL_VARIANTS[variant_name] = None

if MODEL is not None:
    try:
        GRADCAM_MODEL = build_gradcam_model(MODEL)
    except Exception as exc:
        GRADCAM_ERROR = f"Grad-CAM model gagal dibangun: {exc}"
        GRADCAM_MODEL = None


# ─────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────
def normalize_model_probs(probs):
    probs = np.asarray(probs, dtype=np.float32).ravel()
    if probs.shape[0] == 0:
        return probs
    if probs.size > 1 and np.isclose(probs.sum(), 0.0):
        return np.full_like(probs, 1.0 / probs.size, dtype=np.float32)
    probs = probs / np.clip(probs.sum(), 1e-8, None)
    return probs.astype(np.float32)


def summarize_probabilities(probs):
    probs = normalize_model_probs(probs)
    rows = []
    for label, score in zip(CLASS_NAMES, probs):
        rows.append(
            {
                "Kelas": LABEL_LABELS.get(label, label).title(),
                "Label": label,
                "Persentase": float(score * 100.0),
                "Jumlah": float(score),
            }
        )
    return rows


def get_top_label_and_confidence(probs):
    probs = normalize_model_probs(probs)
    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx]
    confidence = float(probs[idx] * 100.0)
    return label, confidence


# ─────────────────────────────────────────────────────────────────────────
# Image loading — HANYA .npy (sesuai data riset/test), JANGAN pakai
# cv2.imdecode() untuk file ini. Fungsi upload PNG/JPG milik
# preprocessing.py (load_image_array_from_bytes, via cv2.imdecode) TIDAK
# dipakai di sini karena tidak sesuai dengan format data yang tersedia.
# ─────────────────────────────────────────────────────────────────────────
def read_npy_scan(uploaded_file):
    """Load X-ray scan dari file .npy yang diunggah pengguna."""
    uploaded_file.seek(0)
    arr = np.load(uploaded_file)
    if arr.ndim == 3 and arr.shape[0] not in (1, 3) and arr.shape[-1] not in (1, 3):
        # volume 3D (mis. beberapa slice) — ambil slice tengah, sama
        # seperti load_image_array() di notebook.
        arr = arr[arr.shape[0] // 2]
    return arr.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────
# Image pipeline — identik dengan preprocessing.py hasil ekspor notebook:
# Gaussian -> ROI crop -> CLAHE/Otsu bone enhancement -> resize -> normalize
# ─────────────────────────────────────────────────────────────────────────
def apply_gaussian_filter(arr, ksize=GAUSSIAN_KERNEL_SIZE, sigma=GAUSSIAN_SIGMA):
    arr_norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.GaussianBlur(arr_norm, ksize, sigma).astype(np.float32)


def crop_lumbar_roi(arr, view="ap"):
    cfg = ROI_CONFIG.get(view, ROI_CONFIG["ap"])
    h, w = arr.shape[:2]
    r0, r1 = int(h * cfg["top"]), int(h * cfg["bottom"])
    c0, c1 = int(w * cfg["left"]), int(w * cfg["right"])
    cropped = arr[r0:r1, c0:c1]
    return cropped if cropped.size > 0 else arr


def enhance_bone_region(arr, clip_limit=2.5, tile_size=(8, 8), mask_blend=0.65):
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


def preprocess_image(scan_array, view="ap"):
    """
    Pipeline: Gaussian -> ROI -> CLAHE/Otsu -> resize -> normalize -> RGB.
    Persis sama dengan notebook. CATATAN PENTING soal skala input:
    di dalam model, layer `rescale_ap`/`rescale_lat` mengalikan input
    dengan 255, lalu backbone EfficientNetB0 punya Rescaling(1/255)
    bawaan sendiri — keduanya saling meniadakan. Artinya model memang
    mengharapkan gambar sudah dinormalisasi ke rentang [0, 1], seperti
    yang dihasilkan fungsi ini (arr/255.0). Jangan panggil
    `resnet50.preprocess_input` atau normalisasi ImageNet mean/std lain
    di sini — itu bukan cara EfficientNetB0 di model ini diberi makan.
    """
    arr = np.asarray(scan_array, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    elif arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(np.asarray(arr, dtype=np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

    arr = apply_gaussian_filter(arr)
    arr = crop_lumbar_roi(arr, view=view)
    arr = enhance_bone_region(arr)
    arr = cv2.resize(arr, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_LINEAR)
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0

    arr = np.stack([arr, arr, arr], axis=-1) if arr.ndim == 2 else arr
    arr = np.expand_dims(arr, axis=0)
    return arr.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────
# Tabular pipeline — WAJIB pakai tab_preprocessor.pkl (StandardScaler +
# OneHotEncoder) yang sama seperti training, BUKAN fitur mentah tanpa
# scaling. PENTING: OneHotEncoder di-fit dengan kategori 'Female'/'Male'
# (title-case) — bukan 'female'/'male' — jadi gender harus dinormalisasi
# ke title-case sebelum ditransformasi, kalau tidak OHE akan meng-nol-kan
# kedua kolom gender (handle_unknown='ignore').
# ─────────────────────────────────────────────────────────────────────────
def prepare_tabular(age, height, weight, bmi, gender):
    gender_title = str(gender).strip().capitalize()  # "female" -> "Female"
    feature_df = pd.DataFrame(
        [{
            "age": float(age),
            "height": float(height),
            "weight": float(weight),
            "bmi": float(bmi),
            "gender": gender_title,
        }]
    )

    if TAB_PREPROCESSOR is None:
        raise RuntimeError(
            "tab_preprocessor.pkl tidak ditemukan. File ini wajib ada supaya "
            "fitur tabular diskalakan (StandardScaler) dan di-encode "
            "(OneHotEncoder) persis seperti saat training."
        )

    transformed = TAB_PREPROCESSOR.transform(feature_df)
    return {
        "full": np.asarray(transformed, dtype=np.float32),
        "raw": feature_df,
    }


# ─────────────────────────────────────────────────────────────────────────
# Prediction — model D (EfficientNetB0 dual-view + gating fusion), satu-
# satunya checkpoint yang tersedia. Input names asli: input_ap, input_lat,
# input_tab.
# ─────────────────────────────────────────────────────────────────────────
def predict_clinical_only(model, tabular):
    pred = model.predict({INPUT_TAB_NAME: tabular["full"]}, verbose=0)
    if isinstance(pred, list):
        pred = pred[0]
    return normalize_model_probs(np.asarray(pred).reshape(-1))


def predict_image_only(model, ap_scan, lat_scan):
    ap_batch = preprocess_image(ap_scan, view="ap")
    lat_batch = preprocess_image(lat_scan, view="lat")
    pred = model.predict({INPUT_AP_NAME: ap_batch, INPUT_LAT_NAME: lat_batch}, verbose=0)
    if isinstance(pred, list):
        pred = pred[0]
    return normalize_model_probs(np.asarray(pred).reshape(-1))


def predict_multimodal(model, ap_scan, lat_scan, tabular):
    ap_batch = preprocess_image(ap_scan, view="ap")
    lat_batch = preprocess_image(lat_scan, view="lat")
    pred = model.predict(
        {
            INPUT_AP_NAME: ap_batch,
            INPUT_LAT_NAME: lat_batch,
            INPUT_TAB_NAME: tabular["full"],
        },
        verbose=0,
    )
    if isinstance(pred, list):
        pred = pred[0]
    return normalize_model_probs(np.asarray(pred).reshape(-1))


def collect_model_confidences(ap_scan, lat_scan, tabular):
    rows = []
    for model_name, model in MODEL_VARIANTS.items():
        if model is None:
            continue
        try:
            if model_name == "Clinical Only":
                probs = predict_clinical_only(model, tabular)
            elif model_name == "Image Only":
                probs = predict_image_only(model, ap_scan, lat_scan)
            elif model_name == "Multimodal No Gating":
                probs = predict_multimodal(model, ap_scan, lat_scan, tabular)
            else:
                probs = predict_multimodal(model, ap_scan, lat_scan, tabular)

            label, confidence = get_top_label_and_confidence(probs)
            rows.append(
                {
                    "Model": model_name,
                    "Top Label": LABEL_LABELS.get(label, label).title(),
                    "Confidence (%)": float(confidence),
                    "Score": float(np.max(normalize_model_probs(probs))),
                }
            )
        except Exception:
            continue

    return sorted(rows, key=lambda x: x["Confidence (%)"], reverse=True)


# ─────────────────────────────────────────────────────────────────────────
# Grad-CAM — implementasi gradient-based yang benar (Selvaraju et al.),
# dihitung pada model gabungan D yang sesungguhnya (bukan sekadar
# rata-rata feature map tanpa gradient). Menggunakan GRADCAM_MODEL yang
# outputnya [conv_ap, conv_lat, output] dalam satu graph yang sama.
# ─────────────────────────────────────────────────────────────────────────
def compute_gradcam_heatmap(gradcam_model, ap_scan, lat_scan, tabular, view="ap", class_idx=0):
    if gradcam_model is None:
        raise ValueError("Grad-CAM model belum tersedia (model utama gagal dimuat).")

    ap_batch = preprocess_image(ap_scan, view="ap")
    lat_batch = preprocess_image(lat_scan, view="lat")

    ap_t = tf.convert_to_tensor(ap_batch)
    lat_t = tf.convert_to_tensor(lat_batch)
    tab_t = tf.convert_to_tensor(tabular["full"])

    with tf.GradientTape() as tape:
        conv_ap, conv_lat, preds = gradcam_model([ap_t, lat_t, tab_t], training=False)
        conv_out = conv_ap if view == "ap" else conv_lat
        tape.watch(conv_out)
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    if grads is None:
        raise ValueError("Gradient Grad-CAM gagal dihitung (grads=None).")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0.0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(original_scan, heatmap, alpha=0.55):
    original = np.asarray(original_scan, dtype=np.float32)
    if original.ndim == 2:
        original = np.stack([original, original, original], axis=-1)
    elif original.ndim == 3 and original.shape[2] == 1:
        original = np.repeat(original, 3, axis=2)

    if original.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)

    if original.max() <= 1.0:
        original_u8 = np.clip(original * 255.0, 0, 255).astype(np.uint8)
    else:
        original_u8 = np.clip(original, 0, 255).astype(np.uint8)

    heatmap_u8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_u8, 1.0 - alpha, heatmap_colored, alpha, 0)
    return overlay


# ─────────────────────────────────────────────────────────────────────────
# UI
# .stApp { background: linear-gradient(180deg, #3e3e75 0%, #45a9a9 100%); }
# ─────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    
    .stApp { background: #3e3e75; }
    .title-box {
        background: linear-gradient(135deg, #0f172a, #3e3e75);
        border-radius: 18px;
        padding: 1.5rem 1.2rem;
        margin-bottom: 1rem;
        color: white;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.12);
    }
    .title-box h1 { margin: 0; font-size: 2.1rem; }
    .title-box p { margin: 0.4rem 0 0; color: #e3f2fd; }
    .result-banner {
        background: #45a9a9;
        border-left: 6px solid #dbe2f0;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid #dbe2f0;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="title-box">
      <h1>🦴 Deteksi Osteoporosis Multimodal</h1>
      <p>Model gabungan citra X-ray (EfficientNetB0) dan data klinis pasien untuk memprediksi status osteoporosis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Tentang model")
    st.write("Aplikasi ini menggunakan checkpoint terbaik (varian D — Multimodal + Gating):")
    st.code("best_D_Multimodal_Gating.keras")
    st.caption("Backbone citra: EfficientNetB0 (dual-view AP + Lateral)")
    st.caption("Fitur klinis: umur, tinggi badan, berat badan, BMI, jenis kelamin (di-scale via tab_preprocessor.pkl)")
    st.info("Prediksi ini bersifat penunjang klinis, bukan diagnosis akhir.")

    if MODEL is not None and st.checkbox("📊 Model Structure Info"):
        st.write(f"**Inputs:** {len(MODEL.inputs)}")
        for inp in MODEL.inputs:
            st.write(f"- Name: `{inp.name}` | Shape: {inp.shape}")

    if MODEL is None or TAB_PREPROCESSOR is None:
        if MODEL is None:
            if MODEL_LOAD_ERROR:
                st.error(MODEL_LOAD_ERROR)
            else:
                st.error(f"Model tidak ditemukan di: {MODEL_PATH}")
        if TAB_PREPROCESSOR is None:
            st.error(f"tab_preprocessor.pkl tidak ditemukan di: {TAB_PREPROCESSOR_PATH}")

        if GRADCAM_ERROR:
            st.warning(GRADCAM_ERROR)

        st.warning(
            "Kalau file-nya SUDAH ada di folder yang benar tapi pesan ini "
            "masih muncul: klik menu kanan-atas (⋮) → **Clear cache**, lalu "
            "**Rerun** — @st.cache_resource bisa menyimpan hasil 'model "
            "tidak ditemukan' dari percobaan sebelumnya dan tidak otomatis "
            "diperbarui hanya dengan menyimpan ulang file."
        )

        with st.expander("🔍 Debug: isi folder yang sebenarnya terbaca aplikasi"):
            st.write(f"`os.getcwd()` (current working directory): `{os.getcwd()}`")
            st.write(f"`BASE_DIR` (folder app.py): `{BASE_DIR}`")
            st.text("Isi cwd/deploy_artifacts:")
            st.code(_debug_dir_listing(Path("deploy_artifacts")))
            st.text("Isi BASE_DIR/deploy_artifacts:")
            st.code(_debug_dir_listing(BASE_DIR / "deploy_artifacts"))

st.subheader("Input Data Pasien")

with st.form("osteoporosis_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Umur", min_value=18, max_value=120, value=55)
        height = st.number_input("Tinggi badan (m)", min_value=1.0, max_value=2.5, value=1.62, step=0.01)
    with c2:
        weight = st.number_input("Berat badan (kg)", min_value=20.0, max_value=200.0, value=65.0, step=0.5)
        gender = st.selectbox("Jenis kelamin", ["female", "male"])
    with c3:
        bmi = weight / (height ** 2)
        st.metric("BMI (Otomatis)", f"{bmi:.2f}")

    st.markdown("---")
    st.write("Upload file X-ray AP dan lateral (.npy) — model D membutuhkan keduanya bersama data klinis.")
    ap_col, lat_col = st.columns(2)
    with ap_col:
        ap_file = st.file_uploader("X-ray AP (.npy)", type=["npy"])
    with lat_col:
        lat_file = st.file_uploader("X-ray Lateral (.npy)", type=["npy"])

    submitted = st.form_submit_button("Prediksi Osteoporosis")

if submitted:
    if MODEL is None:
        st.error("Model tidak tersedia. Pastikan best_D_Multimodal_Gating.keras ada di folder yang sama dengan app.py.")
        st.stop()
    if TAB_PREPROCESSOR is None:
        st.error("tab_preprocessor.pkl tidak tersedia. Fitur tabular tidak bisa diproses dengan benar tanpa file ini.")
        st.stop()
    if ap_file is None or lat_file is None:
        st.warning("⚠️ Model D (Multimodal Gating) membutuhkan X-ray AP dan Lateral (.npy) sekaligus data klinis. Silakan unggah kedua file .npy.")
        st.stop()

    try:
        ap_scan = read_npy_scan(ap_file)
        lat_scan = read_npy_scan(lat_file)
    except Exception as exc:
        st.error(f"File X-ray tidak valid: {exc}")
        st.stop()

    try:
        tabular = prepare_tabular(age, height, weight, bmi, gender)
    except Exception as exc:
        st.error(f"Gagal memproses data klinis: {exc}")
        st.stop()

    try:
        with st.spinner("Menganalisis data klinis dan citra X-ray (Multimodal)..."):
            final_probs = predict_multimodal(MODEL, ap_scan, lat_scan, tabular)
            final_label, final_conf = get_top_label_and_confidence(final_probs)

        st.markdown(
            f"""
            <div class="result-banner">
              <strong>Hasil akhir (Multimodal — Model D):</strong> <span style="font-size:1.5rem; color:#ffffff;">{LABEL_LABELS.get(final_label, final_label).title()}</span>
              <span style="margin-left: 1rem; font-weight:700; color:#ffffff;">Confidence: {final_conf:.2f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.subheader("Confidence per Model")
        variant_rows = collect_model_confidences(ap_scan, lat_scan, tabular)
        if variant_rows:
            comparison_df = pd.DataFrame(variant_rows)
            st.dataframe(
                comparison_df[["Model", "Top Label", "Confidence (%)", "Score"]].rename(
                    columns={
                        "Top Label": "Label Terbaik",
                        "Confidence (%)": "Confidence (%)",
                        "Score": "Probabilitas",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            #st.bar_chart(
            #    comparison_df.set_index("Model")["Confidence (%)"],
            #    use_container_width=True,
            #)
        else:
            st.caption("Tidak ada model pembanding yang bisa diprediksi pada sesi ini.")

        st.markdown("---")
        st.subheader("Grad-CAM untuk Penjelasan Prediksi")
        final_idx = int(np.argmax(final_probs))

        try:
            ap_heatmap = compute_gradcam_heatmap(
                GRADCAM_MODEL, ap_scan, lat_scan, tabular, view="ap", class_idx=final_idx,
            )
            lat_heatmap = compute_gradcam_heatmap(
                GRADCAM_MODEL, ap_scan, lat_scan, tabular, view="lat", class_idx=final_idx,
            )

            ap_overlay = overlay_gradcam(ap_scan, ap_heatmap)
            lat_overlay = overlay_gradcam(lat_scan, lat_heatmap)

            ap_col, lat_col = st.columns(2)
            with ap_col:
                st.caption("AP X-ray + Grad-CAM")
                st.image(ap_overlay, channels="BGR", use_container_width=True)
            with lat_col:
                st.caption("Lateral X-ray + Grad-CAM")
                st.image(lat_overlay, channels="BGR", use_container_width=True)
        except Exception as exc:
            st.warning(f"Grad-CAM tidak dapat dibuat: {exc}")

        st.subheader("Probabilitas per Kelas")
        final_rows = summarize_probabilities(final_probs)
        final_df = pd.DataFrame(final_rows)
        final_df = final_df.sort_values("Persentase", ascending=False).reset_index(drop=True)

        st.dataframe(
            final_df[["Kelas", "Persentase", "Jumlah"]].rename(
                columns={
                    "Kelas": "Kelas",
                    "Persentase": "Confidence (%)",
                    "Jumlah": "Jumlah / score",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        for row in final_rows:
            pct = row["Persentase"]
            st.write(f"**{row['Kelas']}** — {pct:.2f}%")
            st.progress(min(max(pct / 100.0, 0.0), 1.0))

        st.caption("Jumlah / score menampilkan nilai probabilitas normalisasi kelas yang dihasilkan model.")

        if final_conf >= 70:
            st.success("Model sangat yakin dengan hasil prediksi.")
        elif final_conf >= 50:
            st.info("Model cukup yakin, namun tetap disarankan pemeriksaan klinis lanjutan.")
        else:
            st.warning("Model belum terlalu yakin. Pertimbangkan evaluasi ahli atau pemeriksaan tambahan.")

    except Exception as e:
        st.error(f"❌ Kesalahan dalam prediksi model:\n\n{str(e)}")
        st.info("💡 Periksa apakah input X-ray dan data klinis sesuai dengan format yang diharapkan model.")
