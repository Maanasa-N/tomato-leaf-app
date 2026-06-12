import streamlit as st
import numpy as np
import pandas as pd
import cv2
import joblib
import json
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.metrics import pairwise_distances


# ==========================
# 1. LOAD MODELS & METADATA
# ==========================
@st.cache_resource
def load_assets():
    bin_model = joblib.load("models/final_best_binary_model.pkl")
    bin_scaler = joblib.load("models/scaler_binary.pkl")

    multi_model = joblib.load("models/final_best_multiclass_diseased_model.pkl")
    multi_scaler = joblib.load("models/scaler_multiclass_diseased.pkl")

    healthy_centroid = np.load("models/healthy_centroid.npy")

    with open("models/severity_stats.json", "r") as f:
        severity_stats = json.load(f)

    try:
        with open("models/class_id_to_name.json", "r") as f:
            class_id_to_name = json.load(f)
            class_id_to_name = {int(k): v for k, v in class_id_to_name.items()}
    except FileNotFoundError:
        class_id_to_name = {}

    return (
        bin_model, bin_scaler,
        multi_model, multi_scaler,
        healthy_centroid, severity_stats, class_id_to_name
    )


(
    bin_model, bin_scaler,
    multi_model, multi_scaler,
    healthy_centroid, severity_stats, class_id_to_name
) = load_assets()

# ==========================
# 2. FEATURE EXTRACTION
# ==========================

def extract_color_moments(image_hsv):
    features = []
    for channel in cv2.split(image_hsv):
        mean = np.mean(channel)
        std = np.std(channel)
        skewness = np.mean(((channel - mean) ** 3)) / (std ** 3 + 1e-7)
        features.extend([mean, std, skewness])
    return np.array(features)


def extract_color_histogram(image_hsv, bins=16):
    features = []
    for channel in cv2.split(image_hsv):
        hist = cv2.calcHist([channel], [0], None, [bins], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-7)
        features.extend(hist)
    return np.array(features)


def extract_lbp_features(image_gray, P=8, R=1):
    lbp = local_binary_pattern(image_gray, P, R, method='uniform')
    n_bins = P * (P - 1) + 3  # 59
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    lbp_hist = lbp_hist.astype(float) / (lbp_hist.sum() + 1e-7)
    return lbp_hist


def extract_glcm_features(image_gray, distances=[5], angles=[0]):
    glcm = graycomatrix(
        image_gray,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True
    )
    features = [
        graycoprops(glcm, 'contrast').mean(),
        graycoprops(glcm, 'homogeneity').mean(),
        graycoprops(glcm, 'energy').mean(),
        graycoprops(glcm, 'correlation').mean()
    ]
    return np.array(features)


def extract_hu_moments(image_gray):
    _, binary = cv2.threshold(
        image_gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return np.zeros(7)
    cnt = max(contours, key=cv2.contourArea)
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu


def extract_morphological_features(image_gray):
    _, binary = cv2.threshold(
        image_gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return np.zeros(6)
    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = (4 * np.pi * area) / (perimeter**2 + 1e-7)

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / (h + 1e-7)
    extent = area / (w * h + 1e-7)

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-7)

    return np.array([area, perimeter, circularity, aspect_ratio, extent, solidity])

@st.cache_resource
def extract_features_from_bgr_cached(image_bgr):
    return extract_features_from_bgr(image_bgr)

def extract_features_from_bgr(img_bgr):
    """Return 133-D feature vector from a BGR image."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (256, 256))

    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    color_moments = extract_color_moments(img_hsv)
    color_hist = extract_color_histogram(img_hsv, bins=16)
    lbp = extract_lbp_features(img_gray)
    glcm = extract_glcm_features(img_gray)
    hu = extract_hu_moments(img_gray)
    morph = extract_morphological_features(img_gray)

    feats = np.concatenate([color_moments, color_hist, lbp, glcm, hu, morph])
    return feats


# Simple rule: class name containing "healthy" means healthy
def is_class_healthy(name: str) -> bool:
    return "healthy" in name.lower()



# ==========================
# 3. STREAMLIT UI
# ==========================

st.set_page_config(page_title="Tomato Leaf Diagnosis", layout="wide")

st.title("🍅 Tomato Leaf Disease Diagnosis ")
st.markdown(
"""
Upload a tomato leaf image to check if it is healthy or infected.  
If infected, the app identifies the disease and estimates how severe it is.
"""
)
col_left, col_right = st.columns([2.2, 1])

with col_left:
    uploaded_file = st.file_uploader("Upload a leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Could not read the image. Please try again.")
        else:
            img_rgb_display = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb_display, caption="Uploaded Leaf Image", use_container_width=True)

            # extract features once
            feats = extract_features_from_bgr_cached(img_bgr)

            # ========== STAGE 1: BINARY ==========
            st.markdown("### 1️⃣ Health Check")
            if st.button("Check if Healthy"):
                feats_bin = bin_scaler.transform(feats.reshape(1, -1))
                prob_bin = bin_model.predict_proba(feats_bin)[0]   # [p(healthy), p(diseased)]
                pred_bin = int(np.argmax(prob_bin))
                prob_healthy, prob_diseased = prob_bin[0], prob_bin[1]

                st.write(f"**Healthy probability:** {prob_healthy*100:.2f}%")
                st.write(f"**Diseased probability:** {prob_diseased*100:.2f}%")

                is_diseased = (pred_bin == 1)
                st.session_state["is_diseased"] = is_diseased
                st.session_state["feats"] = feats

                if not is_diseased:
                    st.success("The leaf appears **HEALTHY**.")
                else:
                    st.warning("The leaf appears **DISEASED**.")
                    st.info("You can now identify the disease and see severity details.")

            # ========== STAGE 2: MULTICLASS ON DISEASED ==========
            st.markdown("### 2️⃣  Disease Identification")

            if st.button("Identify Disease"):
                if "is_diseased" not in st.session_state:
                    st.error("Please run 'Check if Healthy' first.")
                elif not st.session_state["is_diseased"]:
                    st.info("The leaf is healthy. No disease detected.")
                else:
                    feats = st.session_state["feats"]
                    feats_multi = multi_scaler.transform(feats.reshape(1, -1))
                    probs_multi = multi_model.predict_proba(feats_multi)[0]
                    pred_multi_id = int(np.argmax(probs_multi))

                    disease_name = class_id_to_name.get(pred_multi_id, f"Class {pred_multi_id}")
                    st.session_state["pred_disease_id"] = pred_multi_id

                    st.subheader("Predicted Disease")
                    st.write(f"**Class:** {disease_name}")
                    st.write(f"**Confidence:** {probs_multi[pred_multi_id]*100:.2f}%")


            # ========== STAGE 3: SEVERITY ==========
            st.markdown("### 3️⃣ Estimate Disease Severity")

            if st.button("Estimate Severity"):
                if "is_diseased" not in st.session_state:
                    st.error("Please run 'Check if Healthy' first.")
                elif not st.session_state["is_diseased"]:
                    st.info("Leaf is predicted healthy. Severity is near 0.")
                else:
                    feats = st.session_state["feats"]

                    # use same scaling as binary (healthy centroid is in binary scaled space)
                    feats_bin = bin_scaler.transform(feats.reshape(1, -1))

                    d_min = severity_stats.get("d_min", 0.0)
                    d_max = severity_stats.get("d_max", 1.0)

                    dist = pairwise_distances(
                        feats_bin,
                        healthy_centroid.reshape(1, -1),
                        metric="euclidean"
                    )[0, 0]

                    # normalize to 0–100 using training min/max
                    severity_raw = (dist - d_min) / (d_max - d_min + 1e-7)
                    severity_score = float(np.clip(severity_raw, 0.0, 1.0) * 100.0)

                    if severity_score < 33:
                        level = "Low"
                        color = "green"
                    elif severity_score < 66:
                        level = "Moderate"
                        color = "orange"
                    else:
                        level = "Severe"
                        color = "red"

                    st.markdown(
                        f"<h4 style='color:{color};'>Severity Level: {level}</h4>",
                        unsafe_allow_html=True
                    )
                    st.write(f"Severity Score: **{severity_score:.1f}/100**")
                    st.write("Higher score means stronger infection.")

                    st.markdown(
                        """
                        - Low: Very close to healthy feature patterns  
                        - Moderate: Noticeable deviation, visible symptoms  
                        - Severe: Strong deviation, large infected area / heavy damage  
                        """
                    )

    else:
        st.info("Please upload a tomato leaf image to start.")
# ===============================
# CHATBOT (Right Column Bottom)
# ======================================================
with col_right:
    st.subheader("Tomato Doc 🌱")

    st.write(
        "Ask about the predicted disease, prevention, or treatment. "
    )

    st.markdown("📝 *Examples: 'How to prevent?', 'How to treat?', 'What are symptoms?'*")

    user_q = st.text_area("Your question:")

    if st.button("Ask Chatbot"):

        if not user_q.strip():
            st.info("Please type a question.")
            st.stop()

        disease_name = class_id_to_name.get(
            st.session_state.get("pred_disease_id", None),
            "the disease"
        )

        q_lower = user_q.lower()

        if "prevent" in q_lower:
            answer = (
                f"To prevent **{disease_name}**:\n"
                "- Use disease-free seeds\n"
                "- Keep foliage dry\n"
                "- Remove infected leaves\n"
                "- Rotate crops\n"
            )
        elif "treat" in q_lower or "spray" in q_lower:
            answer = (
                f"For **{disease_name}**, treatments include:\n"
                "- Remove infected leaves\n"
                "- Improve airflow\n"
                "- Use fungicides or bactericides\n"
                "- Organic: neem/copper sprays\n"
            )
        else:
            answer = (
                f"Symptoms of **{disease_name}** often include spots, discoloration.\n"
                "Management includes sanitation, ventilation, and targeted sprays."
            )

        st.success(answer)
