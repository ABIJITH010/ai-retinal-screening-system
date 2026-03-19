from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def apply_colormap_on_image(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    """
    Overlay a heatmap onto an RGB image.

    Args:
        image: PIL RGB image.
        heatmap: numpy array [H, W] in [0,1] (e.g. from Grad-CAM).
        alpha: heatmap intensity (0..1).

    Returns:
        PIL RGB image with heatmap overlay.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    img_np = np.array(image).astype(np.float32) / 255.0

    h, w, _ = img_np.shape
    if heatmap.shape != (h, w):
        logger.info(
            "Resizing heatmap from %s to %s for overlay",
            heatmap.shape,
            (h, w),
        )
        import cv2  # type: ignore

        heatmap = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    heatmap = np.clip(heatmap, 0.0, 1.0)

    # Apply color map (JET-like) using simple RGB mapping
    colored = _simple_jet_colormap(heatmap)

    overlay = (1.0 - alpha) * img_np + alpha * colored
    overlay = np.clip(overlay, 0.0, 1.0)
    overlay_uint8 = (overlay * 255.0).astype(np.uint8)

    return Image.fromarray(overlay_uint8, mode="RGB")


def _simple_jet_colormap(heatmap: np.ndarray) -> np.ndarray:
    """
    Lightweight approximation of JET colormap using numpy only.
    """
    h = np.clip(heatmap, 0.0, 1.0).astype(np.float32)

    r = np.clip(1.5 * h - 0.5, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * h - 1.0), 0.0, 1.0)
    b = np.clip(1.5 * (1.0 - h) - 0.5, 0.0, 1.0)

    colored = np.stack([r, g, b], axis=-1)
    return colored

