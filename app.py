import os
import json

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

MODEL_DIR = "model_lumas_dual"
METADATA_PATH = os.path.join("deploy_artifacts", "metadata.json")

CLASS_NAMES = ["normal", "osteopenia", "osteoporosis"]
LABEL_LABELS = {
    "normal": "Normal",
    "osteopenia": "Osteopenia",
    "osteoporosis": "Osteoporosis",
}


def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


METADATA = load_metadata()


def safe_model_path(filename):
    return os.path.join(MODEL_DIR, filename)


@st.cache_resource
def load_models():
    model_files = {
        "clinical": "best_A_Clinical_Only.keras",
        "image": "best_B_Image_Only.keras",
        "multimodal": "best_D_Multimodal_Gating.keras",
        "multimodal_nogate": "best_C_Multimodal_NoGating.keras",
    }

    models = {}
    for name, file_name in model_files.items():
        path = safe_model_path(file_name)
        if os.path.exists(path):
            models[name] = tf.keras.models.load_model(path, compile=False)
    return models


MODELS = load_models()


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


def read_npy_scan(uploaded_file):
    """Load X-ray scan from .npy file."""
    uploaded_file.seek(0)
    scan_array = np.load(uploaded_file)
    return scan_array


def preprocess_image(scan_array, input_size=(224, 224), backbone="resnet50"):
    """Preprocess X-ray scan (numpy array) for model input."""
    arr = np.asarray(scan_array, dtype=np.float32)
    # Handle grayscale by converting to RGB if needed
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    # Resize to model input size
    arr = cv2.resize(arr, (input_size[1], input_size[0]), interpolation=cv2.INTER_LINEAR)
    if backbone.lower() == "resnet50":
        arr = tf.keras.applications.resnet50.preprocess_input(arr)
    else:
        arr = keras.applications.efficientnet.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr.astype(np.float32)


def build_model_inputs(model, ap_scan, lat_scan, tabular=None):
    ap_batch = preprocess_image(ap_scan, backbone="resnet50")
    lat_batch = preprocess_image(lat_scan, backbone="resnet50")
    input_names = [t.name.split(":")[0] for t in model.inputs]
    if "input_tab" in input_names:
        if tabular is None:
            raise ValueError("Tabular data is required for multimodal model inputs.")
        return [ap_batch, lat_batch, tabular["full"]]
    return [ap_batch, lat_batch]


def compute_gradcam_heatmap(model, ap_scan, lat_scan, tabular=None, branch_name="resnet50_ap", class_idx=0):
    """Create a stable activation heatmap from the final ResNet feature map.

    This is a lightweight Grad-CAM-style explanation that works reliably with the
    saved functional Keras model used in this app. It highlights spatial regions
    with strong activation in the AP or lateral branch.
    """
    if model is None:
        raise ValueError("Model tidak tersedia untuk Grad-CAM.")

    if branch_name not in ["resnet50_ap", "resnet50_lat"]:
        raise ValueError(f"Branch Grad-CAM tidak valid: {branch_name}")

    branch = model.get_layer(branch_name)
    ap_batch = preprocess_image(ap_scan, backbone="resnet50")
    lat_batch = preprocess_image(lat_scan, backbone="resnet50")

    input_tensor = ap_batch if branch_name == "resnet50_ap" else lat_batch
    feature_map = branch(input_tensor, training=False)

    if isinstance(feature_map, list):
        feature_map = feature_map[0]

    feature_map = tf.convert_to_tensor(feature_map, dtype=tf.float32)
    heatmap = tf.reduce_mean(feature_map, axis=-1)
    heatmap = tf.maximum(heatmap, 0.0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap[0].numpy()


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


def prepare_tabular(age, height, weight, bmi, gender):
    # One-hot encode gender: gender_female and gender_male
    gender_female = 1.0 if str(gender).lower() == "female" else 0.0
    gender_male = 1.0 if str(gender).lower() == "male" else 0.0
    
    # Model expects 6 features: age, height, weight, bmi, gender_female, gender_male
    features = np.array(
        [[float(age), float(height), float(weight), float(bmi), gender_female, gender_male]],
        dtype=np.float32,
    )
    return {
        "full": features,
        "legacy": {"age": float(age), "height": float(height), "weight": float(weight), "bmi": float(bmi), "gender": gender},
    }


def get_model_input_info(model):
    """Get model input names and shapes for debugging."""
    input_info = {}
    for inp in model.inputs:
        input_info[inp.name] = inp.shape
    return input_info


def try_predict_payload(model, payloads):
    last_error = None
    for i, payload in enumerate(payloads):
        try:
            pred = model.predict(payload, verbose=0)
            if isinstance(pred, list):
                pred = pred[0]
            pred = np.asarray(pred).reshape(-1)
            return normalize_model_probs(pred)
        except Exception as e:
            last_error = e
            continue
    
    # Debug information
    input_info = get_model_input_info(model)
    error_msg = f"Format input tidak cocok. Model inputs: {input_info}. Last error: {str(last_error)}"
    raise ValueError(error_msg)


def predict_image_only(model, ap_scan, lat_scan):
    """Predict using image model with AP and lateral X-rays."""
    ap_batch = preprocess_image(ap_scan)
    lat_batch = preprocess_image(lat_scan)
    payloads = [
        [ap_batch, lat_batch],
        {"input_ap": ap_batch, "input_lat": lat_batch},
    ]
    return try_predict_payload(model, payloads)


def predict_clinical_only(model, tabular):
    """Predict using clinical model with tabular data."""
    payloads = [
        {"input_tab": tabular["full"]},
        tabular["full"],
    ]
    return try_predict_payload(model, payloads)


def predict_multimodal(model, ap_scan, lat_scan, tabular):
    """Predict using multimodal model with images and tabular data."""
    ap_batch = preprocess_image(ap_scan)
    lat_batch = preprocess_image(lat_scan)
    payloads = [
        {"input_ap": ap_batch, "input_lat": lat_batch, "input_tab": tabular["full"]},
        [ap_batch, lat_batch, tabular["full"]],
    ]
    return try_predict_payload(model, payloads)


def get_top_label_and_confidence(probs):
    probs = normalize_model_probs(probs)
    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx]
    confidence = float(probs[idx] * 100.0)
    return label, confidence


st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #3e3e75 0%, #45a9a9 100%); }
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
        background: #3e3e75;
        border-left: 6px solid #0f172a;
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
      <p>Model gabungan citra X-ray dan data klinis pasien untuk memprediksi status osteoporosis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Tentang model")
    st.write("Aplikasi ini menggunakan model multimodal terbaik untuk osteoporosis:")
    st.code("best_D_Multimodal_Gating.keras")
    st.caption("Fitur klinis: umur, tinggi badan, berat badan, BMI, jenis kelamin")
    st.caption("Fitur citra: X-ray AP & lateral")

    st.info("Prediksi ini bersifat penunjang klinis, bukan diagnosis akhir.")
    
    # Model structure debugging
    if st.checkbox("📊 Model Structure Info"):
        st.subheader("Model Input Structures")
        for name, model in MODELS.items():
            with st.expander(f"{name} model"):
                st.write(f"**Inputs:** {len(model.inputs)}")
                for inp in model.inputs:
                    st.write(f"- Name: `{inp.name}` | Shape: {inp.shape}")

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
        # Calculate BMI automatically
        bmi = weight / (height ** 2)
        st.metric("BMI (Otomatis)", f"{bmi:.2f}")

    st.markdown("---")
    st.write("Upload file X-ray AP dan lateral (.npy)")
    ap_col, lat_col = st.columns(2)
    with ap_col:
        ap_file = st.file_uploader("X-ray AP (.npy)", type=["npy"])
    with lat_col:
        lat_file = st.file_uploader("X-ray Lateral (.npy)", type=["npy"])

    submitted = st.form_submit_button("Prediksi Osteoporosis")

if submitted:
    # Check what data is available
    has_images = ap_file is not None and lat_file is not None
    has_tabular = True  # Always have tabular data from form
    
    if not has_images and not has_tabular:
        st.warning("⚠️ Silakan masukkan data (tabular atau citra X-ray).")
        st.stop()

    # Load images if available
    ap_scan = None
    lat_scan = None
    if has_images:
        try:
            ap_scan = read_npy_scan(ap_file)
            lat_scan = read_npy_scan(lat_file)
        except Exception as exc:
            st.error(f"File X-ray tidak valid: {exc}")
            st.stop()

    tabular = prepare_tabular(age, height, weight, bmi, gender)

    if "clinical" not in MODELS or "image" not in MODELS or "multimodal" not in MODELS:
        st.error("Model osteoporosis belum tersedia di folder model_lumas_dual.")
        st.stop()

    try:
        # Determine which model to use based on available data
        if has_images and has_tabular:
            # Use multimodal model
            with st.spinner("Menganalisis data klinis dan citra X-ray (Multimodal)..."):
                clinical_probs = predict_clinical_only(MODELS["clinical"], tabular)
                image_probs = predict_image_only(MODELS["image"], ap_scan, lat_scan)
                multimodal_probs = predict_multimodal(MODELS["multimodal"], ap_scan, lat_scan, tabular)

                clinical_label, clinical_conf = get_top_label_and_confidence(clinical_probs)
                image_label, image_conf = get_top_label_and_confidence(image_probs)
                multimodal_label, multimodal_conf = get_top_label_and_confidence(multimodal_probs)
        
        elif has_images:
            # Use image model only
            with st.spinner("Menganalisis citra X-ray..."):
                image_probs = predict_image_only(MODELS["image"], ap_scan, lat_scan)
                image_label, image_conf = get_top_label_and_confidence(image_probs)
                
                # Set other predictions to None
                clinical_probs = None
                multimodal_probs = None
                clinical_label = clinical_conf = None
                multimodal_label = multimodal_conf = None
        
        elif has_tabular:
            # Use clinical model only
            with st.spinner("Menganalisis data klinis..."):
                clinical_probs = predict_clinical_only(MODELS["clinical"], tabular)
                clinical_label, clinical_conf = get_top_label_and_confidence(clinical_probs)
                
                # Set other predictions to None
                image_probs = None
                multimodal_probs = None
                image_label = image_conf = None
                multimodal_label = multimodal_conf = None
        
        # Determine final result and label
        if multimodal_probs is not None:
            final_probs = multimodal_probs
            final_label = multimodal_label
            final_conf = multimodal_conf
            model_used = "Multimodal"
        elif image_probs is not None:
            final_probs = image_probs
            final_label = image_label
            final_conf = image_conf
            model_used = "Image Only"
        else:
            final_probs = clinical_probs
            final_label = clinical_label
            final_conf = clinical_conf
            model_used = "Clinical Only"
        
        st.markdown(
            f"""
            <div class="result-banner">
              <strong>Hasil akhir ({model_used}):</strong> <span style="font-size:1.5rem; color:#ffffff;">{LABEL_LABELS.get(final_label, final_label).title()}</span>
              <span style="margin-left: 1rem; font-weight:700; color:#ffffff;">Confidence: {final_conf:.2f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Confidence per Modal")
        cols = []
        col_data = []
        
        if clinical_probs is not None:
            cols.append("Clinical")
            col_data.append(("Clinical", clinical_conf, clinical_label))
        
        if image_probs is not None:
            cols.append("Image")
            col_data.append(("Image", image_conf, image_label))
        
        if multimodal_probs is not None:
            cols.append("Multimodal")
            col_data.append(("Multimodal", multimodal_conf, multimodal_label))
        
        # Display columns for available models
        col_widgets = st.columns(len(col_data)) if col_data else []
        for col_widget, (modal_name, conf, label) in zip(col_widgets, col_data):
            with col_widget:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div style="font-size:0.8rem; color:#64748b;">{modal_name}</div>
                        <div style="font-size:2rem; font-weight:700; color:#0f172a;">{conf:.2f}%</div>
                        <div style="font-size:1rem; color:#1d4ed8;">{LABEL_LABELS.get(label, label).title()}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        if has_images:
            st.markdown("---")
            st.subheader("Grad-CAM untuk Penjelasan Prediksi")
            explain_model = MODELS["multimodal"] if multimodal_probs is not None else MODELS["image"]
            final_idx = int(np.argmax(final_probs))

            try:
                ap_gradcam = compute_gradcam_heatmap(
                    explain_model,
                    ap_scan,
                    lat_scan,
                    tabular=tabular if multimodal_probs is not None else None,
                    branch_name="resnet50_ap",
                    class_idx=final_idx,
                )
                lat_gradcam = compute_gradcam_heatmap(
                    explain_model,
                    ap_scan,
                    lat_scan,
                    tabular=tabular if multimodal_probs is not None else None,
                    branch_name="resnet50_lat",
                    class_idx=final_idx,
                )

                ap_overlay = overlay_gradcam(ap_scan, ap_gradcam)
                lat_overlay = overlay_gradcam(lat_scan, lat_gradcam)

                ap_col, lat_col = st.columns(2)
                with ap_col:
                    st.caption("AP X-ray + Grad-CAM")
                    st.image(ap_overlay, channels="BGR", use_container_width=True)
                with lat_col:
                    st.caption("Lateral X-ray + Grad-CAM")
                    st.image(lat_overlay, channels="BGR", use_container_width=True)
            except Exception as exc:
                st.warning(f"Grad-CAM tidak dapat dibuat untuk model saat ini: {exc}")

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
    
    except ValueError as e:
        st.error(f"❌ Kesalahan dalam prediksi model:\n\n{str(e)}")
        st.info("💡 Periksa apakah input X-ray dan data klinis sesuai dengan format yang diharapkan model.")
        st.error(f"❌ Kesalahan dalam prediksi model:\n\n{str(e)}")
        st.info("💡 Periksa apakah input X-ray dan data klinis sesuai dengan format yang diharapkan model.")
