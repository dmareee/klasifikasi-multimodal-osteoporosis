"""
LUMOS — Osteoporosis Risk Classification Dashboard
Deploy: streamlit run streamlit_app.py

Folder yang harus ada di samping file ini (hasil dari cell save-deploy notebook):
    deploy_artifacts/
        best_<variant>.keras
        tab_preprocessor.pkl
        metadata.json
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from tensorflow import keras

from preprocessing import preprocess_uploaded_image, compute_bmi

# ══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="LUMOS — Osteoporosis Risk Classifier",
                    page_icon="🦴", layout="wide")

DEPLOY_DIR = Path("deploy_artifacts")
CLASS_COLORS = {"normal": "#2ecc71", "osteopenia": "#f39c12", "osteoporosis": "#e74c3c"}

# Preprocessing (apply_gaussian_filter, crop_lumbar_roi, enhance_bone_region,
# preprocess_uploaded_image) di-import dari preprocessing.py — modul yang sama
# di-export dari notebook, jadi selalu konsisten dengan pipeline training.


# ══════════════════════════════════════════════════════════════════════════
#  LOAD ARTIFACTS (cached — hanya load sekali per session)
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    with open(DEPLOY_DIR / "metadata.json") as f:
        meta = json.load(f)
    with open(DEPLOY_DIR / "tab_preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    model = keras.models.load_model(DEPLOY_DIR / meta["best_variant_checkpoint"])
    return meta, preprocessor, model


try:
    meta, preprocessor, model = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"❌ Artefak deployment tidak ditemukan: {e}\n\n"
        f"Pastikan folder `deploy_artifacts/` (berisi .keras, tab_preprocessor.pkl, "
        f"metadata.json) ada di direktori yang sama dengan streamlit_app.py."
    )
    st.stop()

variant = meta["best_variant"]
vinfo = meta["variant_config"][variant]
uses_img = vinfo["img"]
uses_tab = vinfo["tab"]
class_names = meta["class_names"]
img_size = meta["img_size"]

# ══════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════
st.title("🦴 LUMOS — Osteoporosis Risk Classification")
st.caption(f"Model aktif: **{variant}**  |  Input: "
           f"{'Citra X-ray (AP + Lateral)' if uses_img else ''}"
           f"{' + ' if uses_img and uses_tab else ''}"
           f"{'Data klinis' if uses_tab else ''}")

col_input, col_result = st.columns([1, 1.2])

with col_input:
    st.subheader("📥 Input Data Pasien")

    ap_file = lat_file = None
    if uses_img:
        c1, c2 = st.columns(2)
        with c1:
            ap_file = st.file_uploader("X-ray AP (Anteroposterior)", type=["png", "jpg", "jpeg"])
            if ap_file:
                st.image(ap_file, caption="AP view", use_container_width=True)
        with c2:
            lat_file = st.file_uploader("X-ray Lateral", type=["png", "jpg", "jpeg"])
            if lat_file:
                st.image(lat_file, caption="Lateral view", use_container_width=True)

    tab_values = {}
    if uses_tab:
        st.markdown("**Data Klinis**")
        c1, c2 = st.columns(2)
        with c1:
            tab_values["age"] = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=60)
            tab_values["height"] = st.number_input("Tinggi badan (cm)", min_value=50.0, max_value=250.0, value=160.0)
        with c2:
            tab_values["weight"] = st.number_input("Berat badan (kg)", min_value=10.0, max_value=250.0, value=60.0)
            tab_values["gender"] = st.selectbox("Jenis kelamin", ["male", "female"])
        tab_values["bmi"] = compute_bmi(tab_values["weight"], tab_values["height"])
        st.caption(f"BMI otomatis dihitung: **{tab_values['bmi']}**")

    predict_btn = st.button("🔍 Prediksi", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ══════════════════════════════════════════════════════════════════════════
with col_result:
    st.subheader("📊 Hasil Klasifikasi")

    if predict_btn:
        # Validasi input
        if uses_img and (ap_file is None or lat_file is None):
            st.warning("⚠️ Upload kedua citra X-ray (AP dan Lateral) dulu.")
            st.stop()

        with st.spinner("Menjalankan preprocessing dan inferensi model..."):
            model_inputs = []

            if uses_img:
                ap_arr = preprocess_uploaded_image(ap_file.getvalue(), target_size=tuple(img_size), view="ap")
                lat_arr = preprocess_uploaded_image(lat_file.getvalue(), target_size=tuple(img_size), view="lat")
                model_inputs.append(np.expand_dims(ap_arr, axis=0))
                model_inputs.append(np.expand_dims(lat_arr, axis=0))

            if uses_tab:
                tab_df = pd.DataFrame([tab_values])
                cols_needed = meta["num_features"] + meta["cat_features"]
                X_tab = preprocessor.transform(tab_df[cols_needed]).astype(np.float32)
                model_inputs.append(X_tab)

            # Kalau cuma 1 modalitas, jangan bungkus list
            final_input = model_inputs[0] if len(model_inputs) == 1 else model_inputs
            probs = model.predict(final_input, verbose=0)[0]

        pred_idx = int(np.argmax(probs))
        pred_class = class_names[pred_idx]
        pred_conf = float(probs[pred_idx])

        # ── Kartu hasil utama ──────────────────────────────────────────
        color = CLASS_COLORS.get(pred_class, "#3498db")
        st.markdown(
            f"""
            <div style="padding:20px; border-radius:12px; background-color:{color}22;
                        border:2px solid {color}; text-align:center;">
                <div style="font-size:14px; color:#555;">PREDIKSI</div>
                <div style="font-size:32px; font-weight:800; color:{color}; text-transform:capitalize;">
                    {pred_class}
                </div>
                <div style="font-size:18px; color:#333;">Confidence: <b>{pred_conf*100:.1f}%</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Bar chart confidence semua kelas ──────────────────────────
        fig = go.Figure(go.Bar(
            x=[float(p) * 100 for p in probs],
            y=[c.capitalize() for c in class_names],
            orientation="h",
            marker_color=[CLASS_COLORS.get(c, "#3498db") for c in class_names],
            text=[f"{p*100:.1f}%" for p in probs],
            textposition="outside",
        ))
        fig.update_layout(
            title="Distribusi Confidence per Kelas",
            xaxis_title="Probabilitas (%)",
            xaxis_range=[0, 100],
            height=280,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Tabel detail ────────────────────────────────────────────────
        detail_df = pd.DataFrame({
            "Kelas": [c.capitalize() for c in class_names],
            "Probabilitas": [f"{p*100:.2f}%" for p in probs],
        }).sort_values("Probabilitas", ascending=False)
        st.dataframe(detail_df, hide_index=True, use_container_width=True)

        if pred_class == "osteoporosis" and pred_conf > 0.7:
            st.error("⚠️ Risiko tinggi terdeteksi. Disarankan konsultasi lanjut dengan dokter spesialis.")
        elif pred_class == "osteopenia":
            st.warning("ℹ️ Terindikasi osteopenia (pra-osteoporosis). Pemantauan berkala disarankan.")
    else:
        st.info("Isi data di panel kiri lalu klik **Prediksi** untuk melihat hasil klasifikasi.")

st.divider()
st.caption(
    f"Model: Dual-View ResNet50 + Tabular MLP ({variant}) · "
    f"Dataset training: {meta.get('dataset_size', '-')} pasien · "
    "LUMOS Osteoporosis Risk Prediction System"
)
