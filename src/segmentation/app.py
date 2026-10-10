import os
import time
import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from src.segmentation.unet import UNet

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "best_improved_tumor_unet.pth")
IMAGE_SIZE = 256

st.set_page_config(page_title="LiverAI Studio", page_icon="🧬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root { --bg:#080b16; --border:#27304b; --cyan:#67e8f9; --muted:#9aa8bf; }
html, body, [class*="css"] { font-family:Inter,sans-serif; }
.stApp {
 background:radial-gradient(ellipse at 12% 0%,rgba(124,58,237,.20),transparent 38%),
 radial-gradient(ellipse at 90% 8%,rgba(34,211,238,.10),transparent 34%),#080b16;
 color:#f1f5f9;
}
[data-testid="stHeader"] {background:rgba(8,11,22,.75);}
[data-testid="stSidebar"] {background:#0b1020;border-right:1px solid var(--border);}
[data-testid="stSidebar"] * {color:#e2e8f0;}
.brand {font-size:27px;font-weight:800;letter-spacing:-1px;color:#f8fafc;}
.brand span {color:var(--cyan);}
.eyebrow {color:var(--cyan);font-size:11px;font-weight:700;letter-spacing:3px;}
.hero-title {font-size:clamp(32px,4vw,49px);font-weight:800;line-height:1.15;letter-spacing:-2px;margin:8px 0;color:#f8fafc;}
.gradient-text {background:linear-gradient(90deg,#a78bfa,#67e8f9);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.subtitle {color:var(--muted);font-size:14px;line-height:1.8;}
.panel {background:linear-gradient(145deg,rgba(23,30,54,.96),rgba(14,19,36,.96));border:1px solid var(--border);border-radius:17px;padding:20px;margin-bottom:14px;}
.panel-label {color:var(--muted);font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;}
.panel-value {font-size:28px;font-weight:800;margin-top:10px;color:#f8fafc;}
.panel-description {color:var(--muted);font-size:12px;margin-top:6px;}
.section-heading {font-size:23px;font-weight:750;color:#f8fafc;margin:18px 0 5px;}
.section-description {color:var(--muted);font-size:13px;margin-bottom:18px;}
.status-pill {display:inline-block;padding:7px 12px;border:1px solid rgba(103,232,249,.35);border-radius:30px;background:rgba(103,232,249,.08);color:#67e8f9;font-size:12px;font-weight:700;}
.stButton>button {width:100%;min-height:44px;border:1px solid #8b5cf6;border-radius:10px;background:linear-gradient(100deg,#7c3aed,#4f46e5);color:white;font-weight:700;}
.stButton>button:hover {border-color:var(--cyan);color:white;box-shadow:0 0 20px rgba(103,232,249,.18);}
.stDownloadButton>button {width:100%;border-radius:10px;border:1px solid var(--border);background:#151c32;color:#e2e8f0;}
[data-testid="stFileUploader"] {background:rgba(17,22,41,.7);border:1px dashed #6250a8;border-radius:14px;padding:12px;}
[data-testid="stMetric"] {background:rgba(17,22,41,.85);border:1px solid var(--border);border-radius:13px;padding:14px;}
[data-testid="stMetricLabel"] {color:var(--muted);}
[data-testid="stMetricValue"] {color:var(--cyan);}
hr {border-color:var(--border);} footer,#MainMenu {visibility:hidden;}

/* Black upload button: Streamlit 1.64 */
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button[kind="secondary"] {
    background: #000000 !important;
    color: #ffffff !important;
    border: 1px solid #67e8f9 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}
[data-testid="stFileUploader"] button p,
[data-testid="stFileUploader"] button span,
[data-testid="stFileUploader"] button svg {
    color: #ffffff !important;
    fill: #ffffff !important;
}
[data-testid="stFileUploader"] button:hover {
    background: #171717 !important;
    border-color: #a78bfa !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model, device


def grayscale(image):
    image = np.asarray(image)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY if image.shape[2] == 4 else cv2.COLOR_RGB2GRAY)
    return image


def enhance(image):
    image = grayscale(image)
    image = np.nan_to_num(image).astype(np.float32)
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip((image - low) / (high - low) * 255, 0, 255).astype(np.uint8)


def preprocess(image):
    image = grayscale(image)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    return image, torch.from_numpy(image).unsqueeze(0).unsqueeze(0)


def clean_mask(mask, min_area):
    binary = (mask > 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 1
    return cleaned.astype(np.float32)


def predict(model, device, tensor):
    tensor = tensor.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        liver_logits, tumor_logits = model(tensor)
        liver_probability = torch.sigmoid(liver_logits)
        tumor_probability = torch.sigmoid(tumor_logits)
        liver_mask = (liver_probability > 0.5).float()
        tumor_mask = (tumor_probability > 0.5).float()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    lp = liver_probability.cpu().numpy()[0, 0]
    tp = tumor_probability.cpu().numpy()[0, 0]
    liver = clean_mask(liver_mask.cpu().numpy()[0, 0], 500)
    tumor = clean_mask(tumor_mask.cpu().numpy()[0, 0], 30)
    return lp, tp, liver, tumor, elapsed


def overlay_image(image, liver, tumor):
    base = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    result = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    result[liver > 0.5] = [0, 220, 150]
    result[tumor > 0.5] = [255, 65, 110]
    return result


def mask_rgb(mask, color):
    result = np.zeros((*mask.shape, 3), dtype=np.uint8)
    result[mask > 0.5] = color
    return result


def png_bytes(array):
    ok, encoded = cv2.imencode(".png", array)
    if not ok:
        raise RuntimeError("Could not encode result image.")
    return encoded.tobytes()


with st.sidebar:
    st.markdown('<div class="brand">◈ Liver<span>AI</span></div>', unsafe_allow_html=True)
    st.caption("MEDICAL IMAGING STUDIO")
    st.divider()
    page = st.radio("NAVIGATION", ["Dashboard", "Scan Analysis", "About Model"])
    st.divider()
    st.markdown("**SYSTEM STATUS**")
    st.markdown("🟢 Segmentation workspace")
    st.caption("U-Net · PyTorch")
    st.caption("Research prototype · Not for diagnosis")

if not os.path.exists(MODEL_PATH):
    st.error("Trained model checkpoint not found.")
    st.code(MODEL_PATH)
    st.stop()

try:
    model, device = load_model()
except Exception as error:
    st.error(f"Could not load model: {error}")
    st.stop()

st.markdown('<div class="eyebrow">INTELLIGENT MEDICAL IMAGE ANALYSIS</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">LiverAI <span class="gradient-text">Studio</span></div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Explore liver and tumor segmentation with deep learning. Upload a CT slice and visualize the model predictions.</p>', unsafe_allow_html=True)
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown('<span class="status-pill">● MODEL LOADED</span>', unsafe_allow_html=True)
with h2:
    st.caption(f"Compute device: {str(device).upper()}")

if page == "Dashboard":
    st.markdown('<div class="section-heading">Welcome to your AI workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-description">A research workspace for experimental liver and tumor segmentation.</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, label, value, desc in zip(cols, ["ARCHITECTURE", "INPUT MODALITY", "OUTPUT CLASSES"], ["U-Net", "CT Scan", "02"], ["Encoder-decoder network", "Single-slice image", "Liver and tumor masks"]):
        with col:
            st.markdown(f'<div class="panel"><div class="panel-label">{label}</div><div class="panel-value">{value}</div><div class="panel-description">{desc}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Analysis workflow</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, number, title, desc in zip(cols, ["01", "02", "03"], ["Upload", "Analyze", "Visualize"], ["Choose a supported CT slice.", "Run the trained U-Net model.", "Inspect and download predicted masks."]):
        with col:
            st.markdown(f'<div class="panel"><div class="panel-value">{number}</div><h4>{title}</h4><p class="subtitle">{desc}</p></div>', unsafe_allow_html=True)
    st.info("Choose **Scan Analysis** in the sidebar to upload an image and run segmentation.")

elif page == "Scan Analysis":
    st.markdown('<div class="section-heading">CT Scan Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-description">Upload a PNG or JPEG CT slice to generate a prediction.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose a CT image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        try:
            pil_image = Image.open(uploaded).convert("RGB")
            raw = np.asarray(pil_image)
            display_image = enhance(raw)
            model_image, tensor = preprocess(raw)
            a, b = st.columns([1.3, 1])
            with a:
                st.image(display_image, caption="Contrast-enhanced input preview", use_container_width=True)
            with b:
                st.markdown("#### Scan information")
                st.metric("Original dimensions", f"{pil_image.width} × {pil_image.height}")
                st.metric("Model input", f"{IMAGE_SIZE} × {IMAGE_SIZE}")
                st.caption("Input is converted to grayscale, resized, and normalized.")
            if st.button("⚡ Run AI Segmentation", type="primary"):
                with st.spinner("Running U-Net segmentation..."):
                    lp, tp, liver, tumor, elapsed = predict(model, device, tensor)
                    st.session_state["liverai_result"] = {
                        "name": uploaded.name, "image": model_image, "liver": liver,
                        "tumor": tumor, "overlay": overlay_image(model_image, liver, tumor),
                        "elapsed": elapsed,
                    }
            result = st.session_state.get("liverai_result")
            if result is not None and result["name"] == uploaded.name:
                st.divider()
                st.markdown('<div class="section-heading">Segmentation viewer</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-description">Teal indicates predicted liver pixels; pink-red indicates predicted tumor pixels.</div>', unsafe_allow_html=True)
                original = (np.clip(result["image"], 0, 1) * 255).astype(np.uint8)
                v1, v2 = st.columns(2)
                v3, v4 = st.columns(2)
                with v1:
                    st.image(original, caption="Original model input", use_container_width=True)
                with v2:
                    st.image(mask_rgb(result["liver"], [0, 220, 150]), caption="Predicted liver mask", use_container_width=True)
                with v3:
                    st.image(mask_rgb(result["tumor"], [255, 65, 110]), caption="Predicted tumor mask", use_container_width=True)
                with v4:
                    st.image(result["overlay"], caption="Segmentation overlay", use_container_width=True)
                liver_pixels = int(np.sum(result["liver"] > 0.5))
                tumor_pixels = int(np.sum(result["tumor"] > 0.5))
                total = IMAGE_SIZE * IMAGE_SIZE
                st.markdown('<div class="section-heading">Prediction statistics</div>', unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Liver pixels", f"{liver_pixels:,}")
                m2.metric("Liver area", f"{100*liver_pixels/total:.2f}%")
                m3.metric("Tumor pixels", f"{tumor_pixels:,}")
                m4.metric("Tumor area", f"{100*tumor_pixels/total:.2f}%")
                st.caption(f"Inference time: {result['elapsed']:.3f} seconds. Timing varies by hardware.")
                st.markdown('<div class="section-heading">Download results</div>', unsafe_allow_html=True)
                d1, d2, d3 = st.columns(3)
                with d1:
                    st.download_button("↓ Liver mask", png_bytes((result["liver"]*255).astype(np.uint8)), "predicted_liver_mask.png", "image/png")
                with d2:
                    st.download_button("↓ Tumor mask", png_bytes((result["tumor"]*255).astype(np.uint8)), "predicted_tumor_mask.png", "image/png")
                with d3:
                    st.download_button("↓ Overlay", png_bytes(cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR)), "segmentation_overlay.png", "image/png")
                st.warning("Research prototype only. Predictions may be wrong and must not be used for medical diagnosis or treatment.")
        except Exception as error:
            st.error(f"Could not process this image: {error}")
    else:
        st.info("Upload a CT image to begin analysis.")

elif page == "About Model":
    st.markdown('<div class="section-heading">About the AI model</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-description">Technical overview of this segmentation research prototype.</div>', unsafe_allow_html=True)
    for title, desc in [
        ("U-Net architecture", "A convolutional encoder-decoder network with skip connections. It produces separate liver and tumor predictions."),
        ("Dataset", "Developed using prepared CT slices from the Liver Tumor Segmentation (LiTS) dataset."),
        ("Output interpretation", "Pixel counts refer to the resized 256 × 256 image, not physical anatomical volume."),
        ("Research limitations", "This educational prototype is not clinically validated. Predictions may contain false positives and false negatives and must not be used for medical decisions."),
    ]:
        st.markdown(f'<div class="panel"><h3>{title}</h3><p class="subtitle">{desc}</p></div>', unsafe_allow_html=True)

st.divider()
st.markdown('<div style="text-align:center;padding:8px 0 22px;"><div class="brand" style="font-size:20px;">◈ Liver<span>AI</span> Studio</div><p style="color:#64748b;font-size:12px;">Deep Learning · Medical Image Segmentation · Research Prototype</p></div>', unsafe_allow_html=True)
