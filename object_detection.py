#!/usr/bin/env python3
"""Run object detection on an image using a YOLOv8 model from HuggingFace.

The script downloads the YOLOv8n model weights from the HuggingFace Hub the
first time it is executed. It then detects objects in an input image and
saves an annotated image showing bounding boxes, class names and confidence
scores.
"""

from __future__ import annotations

import argparse

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO


def load_model() -> YOLO:
    """Download and return the YOLOv8 model."""
    weights_path = hf_hub_download("ultralytics/YOLOv8", "yolov8n.pt")
    return YOLO(weights_path)


def run_detection(model: YOLO, image_path: str) -> np.ndarray:
    """Run object detection and return an annotated RGB image as ``numpy`` array."""
    results = model(image_path)[0]
    annotated_bgr = results.plot()
    annotated_rgb = annotated_bgr[..., ::-1]  # BGR -> RGB
    return annotated_rgb


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 object detection")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "-o",
        "--output",
        default="result.jpg",
        help="Path to save annotated image (default: result.jpg)",
    )
    args = parser.parse_args()

    model = load_model()
    annotated = run_detection(model, args.image)
    Image.fromarray(annotated).save(args.output)
    print(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()
