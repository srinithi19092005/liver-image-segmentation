import os
import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image

from unet import UNet


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "checkpoints",
    "best_improved_tumor_unet.pth",
)

IMAGE_SIZE = 256


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Liver & Tumor Segmentation",
    page_icon="🩺",
    layout="wide",
)


# ============================================================
# CUSTOM STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🩺 Liver & Tumor Segmentation</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    AI-based liver and tumor segmentation from CT images
    using a U-Net deep learning model.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = UNet().to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model, device


# ============================================================
# IMAGE FUNCTIONS
# ============================================================

def convert_to_grayscale(image):

    image = np.array(image)

    if len(image.shape) == 3:

        if image.shape[2] == 4:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGBA2GRAY,
            )

        else:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2GRAY,
            )

    return image


def enhance_for_display(image):

    image = convert_to_grayscale(image)

    image = image.astype(np.uint8)

    low = np.percentile(
        image,
        1,
    )

    high = np.percentile(
        image,
        99,
    )

    if high <= low:
        return image

    enhanced = np.clip(
        (
            image.astype(np.float32)
            - low
        )
        / (high - low)
        * 255.0,
        0,
        255,
    )

    return enhanced.astype(np.uint8)


def preprocess_for_model(image):

    image = convert_to_grayscale(image)

    image = cv2.resize(
        image,
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )

    image = (
        image.astype(np.float32)
        / 255.0
    )

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0).unsqueeze(0)

    return image, tensor


# ============================================================
# REMOVE SMALL COMPONENTS
# ============================================================

def clean_mask(mask, min_area):

    binary = (
        mask > 0.5
    ).astype(np.uint8)

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    cleaned = np.zeros_like(binary)

    for label in range(
        1,
        num_labels,
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA,
        ]

        if area >= min_area:

            cleaned[
                labels == label
            ] = 1

    return cleaned.astype(
        np.float32
    )


# ============================================================
# PREDICTION
# ============================================================

def predict(
    model,
    device,
    image_tensor,
):

    image_tensor = image_tensor.to(
        device
    )

    with torch.no_grad():

        liver_logits, tumor_logits = (
            model(image_tensor)
        )

        liver_probability = torch.sigmoid(
            liver_logits
        )

        tumor_probability = torch.sigmoid(
            tumor_logits
        )

        liver_mask = (
            liver_probability > 0.5
        ).float()

        tumor_mask = (
            tumor_probability > 0.5
        ).float()

    liver_mask = clean_mask(
        liver_mask.cpu().numpy()[0, 0],
        min_area=500,
    )

    tumor_mask = clean_mask(
        tumor_mask.cpu().numpy()[0, 0],
        min_area=30,
    )

    return (
        liver_probability.cpu().numpy()[0, 0],
        tumor_probability.cpu().numpy()[0, 0],
        liver_mask,
        tumor_mask,
    )


# ============================================================
# OVERLAY
# ============================================================

def create_overlay(
    image,
    liver_mask,
    tumor_mask,
):

    image_uint8 = (
        image * 255
    ).astype(np.uint8)

    overlay = cv2.cvtColor(
        image_uint8,
        cv2.COLOR_GRAY2RGB,
    )

    liver_pixels = (
        liver_mask > 0.5
    )

    tumor_pixels = (
        tumor_mask > 0.5
    )

    overlay[
        liver_pixels
    ] = [0, 255, 0]

    overlay[
        tumor_pixels
    ] = [255, 0, 0]

    return overlay


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        "Final model checkpoint was not found."
    )

    st.code(
        MODEL_PATH
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model, device = load_model()

    st.success(
        f"Model loaded successfully | Device: {device}"
    )

except Exception as e:

    st.error(
        f"Model loading error: {e}"
    )

    st.stop()


# ============================================================
# UPLOAD
# ============================================================

st.subheader(
    "1. Upload CT Image"
)

uploaded_file = st.file_uploader(
    "Choose a CT slice",
    type=[
        "png",
        "jpg",
        "jpeg",
    ],
)


# ============================================================
# PROCESS
# ============================================================

if uploaded_file is not None:

    try:

        uploaded_image = Image.open(
            uploaded_file
        )

        raw_image = np.array(
            uploaded_image
        )

        display_image = (
            enhance_for_display(
                raw_image
            )
        )

        model_image, image_tensor = (
            preprocess_for_model(
                raw_image
            )
        )

        # ----------------------------------------------------
        # INPUT IMAGE
        # ----------------------------------------------------

        st.subheader(
            "2. Input CT Image"
        )

        st.image(
            display_image,
            caption="Enhanced CT visualization",
            width="stretch",
        )

        st.caption(
            "Contrast enhancement is used only for "
            "visualization. The original normalized image "
            "is provided to the trained model."
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        with st.spinner(
            "Running U-Net segmentation..."
        ):

            (
                liver_probability,
                tumor_probability,
                liver_mask,
                tumor_mask,
            ) = predict(
                model,
                device,
                image_tensor,
            )

        # ----------------------------------------------------
        # OVERLAY
        # ----------------------------------------------------

        overlay = create_overlay(
            model_image,
            liver_mask,
            tumor_mask,
        )

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        st.subheader(
            "3. Segmentation Results"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.image(
                liver_mask,
                caption="Predicted Liver Mask",
                width="stretch",
            )

        with col2:

            st.image(
                tumor_mask,
                caption="Predicted Tumor Mask",
                width="stretch",
            )

        with col3:

            st.image(
                overlay,
                caption="Green = Liver | Red = Tumor",
                width="stretch",
            )

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        st.subheader(
            "4. Segmentation Statistics"
        )

        liver_pixels = int(
            np.sum(
                liver_mask > 0.5
            )
        )

        tumor_pixels = int(
            np.sum(
                tumor_mask > 0.5
            )
        )

        total_pixels = (
            IMAGE_SIZE * IMAGE_SIZE
        )

        liver_area = (
            liver_pixels
            / total_pixels
            * 100
        )

        tumor_area = (
            tumor_pixels
            / total_pixels
            * 100
        )

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        with col1:

            st.metric(
                "Liver Pixels",
                f"{liver_pixels:,}",
            )

        with col2:

            st.metric(
                "Liver Area",
                f"{liver_area:.2f}%",
            )

        with col3:

            st.metric(
                "Tumor Pixels",
                f"{tumor_pixels:,}",
            )

        with col4:

            st.metric(
                "Tumor Area",
                f"{tumor_area:.2f}%",
            )

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        st.subheader(
            "5. Download Results"
        )

        liver_png = (
            liver_mask * 255
        ).astype(np.uint8)

        tumor_png = (
            tumor_mask * 255
        ).astype(np.uint8)

        overlay_png = cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR,
        )

        _, liver_buffer = cv2.imencode(
            ".png",
            liver_png,
        )

        _, tumor_buffer = cv2.imencode(
            ".png",
            tumor_png,
        )

        _, overlay_buffer = cv2.imencode(
            ".png",
            overlay_png,
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.download_button(
                "⬇️ Liver Mask",
                data=liver_buffer.tobytes(),
                file_name="predicted_liver_mask.png",
                mime="image/png",
                width="stretch",
            )

        with col2:

            st.download_button(
                "⬇️ Tumor Mask",
                data=tumor_buffer.tobytes(),
                file_name="predicted_tumor_mask.png",
                mime="image/png",
                width="stretch",
            )

        with col3:

            st.download_button(
                "⬇️ Overlay",
                data=overlay_buffer.tobytes(),
                file_name="segmentation_overlay.png",
                mime="image/png",
                width="stretch",
            )

    except Exception as e:

        st.error(
            f"Error processing the CT image: {e}"
        )

else:

    st.info(
        "Upload a CT image above to begin segmentation."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Liver and Tumor Segmentation | "
    "U-Net | LiTS Dataset"
)
