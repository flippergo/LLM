#!/usr/bin/env python3
"""Streamlit app for object detection using a YOLOv8 model from HuggingFace.

Upload an image, view the detection results and optionally download the
annotated image.
"""

from __future__ import annotations

from io import BytesIO

import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO


@st.cache_resource
def load_model() -> YOLO:
    """Download and return the YOLOv8 model."""
    weights_path = hf_hub_download("ultralytics/YOLOv8", "yolov8n.pt")
    return YOLO(weights_path)


def run_detection(model: YOLO, image: Image.Image) -> Image.Image:
    """Run object detection and return an annotated RGB ``PIL.Image``."""
    results = model(image)[0]
    annotated_bgr = results.plot()
    annotated_rgb = annotated_bgr[..., ::-1]
    return Image.fromarray(annotated_rgb)


def main() -> None:
    st.title("YOLOv8 Object Detection")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        st.write("Please upload an image file.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect objects"):
        model = load_model()
        annotated = run_detection(model, image)
        st.image(annotated, caption="Detection Result", use_column_width=True)

        buf = BytesIO()
        annotated.save(buf, format="JPEG")
        st.download_button(
            label="Download result",
            data=buf.getvalue(),
            file_name="result.jpg",
            mime="image/jpeg",
        )


if __name__ == "__main__":
    main()
