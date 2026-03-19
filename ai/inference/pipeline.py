from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Union

from PIL import Image
import numpy as np

from .predictor import DRPredictor, PredictorConfig, format_output, _risk_level_from_severity  # noqa: F401

logger = logging.getLogger(__name__)


ImageInput = Union[str, Path, Image.Image, np.ndarray]


def run_inference(
    image: ImageInput,
    model_path: Path | None = None,
) -> Dict[str, Any]:
    """
    High-level inference pipeline:
      image -> preprocess -> model -> prediction JSON

    Returns:
        {
            "prediction": "Moderate DR",
            "confidence": 0.87,
            "severity_level": 2,
            "risk_level": "Medium",
        }
    """
    logger.info("Running DR inference pipeline...")

    cfg = PredictorConfig()
    if model_path is not None:
        cfg.model_path = model_path

    predictor = DRPredictor(cfg)
    result = predictor.predict(image)

    logger.info("Inference pipeline finished. result=%s", result)
    return result

