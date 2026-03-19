from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image

from ai.model.efficientnet import (
    DR_CLASSES,
    build_dr_model,
    get_class_label,
    infer_model_family_from_state_dict,
)
from ai.preprocessing.pipeline import preprocess_for_inference


logger = logging.getLogger(__name__)


ImageInput = Union[str, Path, Image.Image, np.ndarray]


def _lazy_import_torch():
    """
    Lazily import torch and torch.nn.functional.
    This keeps module import safe even if torch is not installed yet.
    """
    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
    except Exception as e:  # pragma: no cover
        logger.error("PyTorch is required for inference: %s", str(e))
        raise RuntimeError(
            "PyTorch is required to run inference. "
            "Please ensure it is installed and configured correctly."
        ) from e
    return torch, F


@dataclass
class PredictorConfig:
    model_path: Path = Path("ai") / "model" / "weights" / "dr_model.pth"
    device: Optional[str] = None  # 'cpu' or 'cuda'; None -> auto
    model_name: str = "ensemble"


def _risk_level_from_severity(severity: int) -> str:
    if severity <= 0:
        return "Low"
    if severity <= 2:
        return "Medium"
    return "High"


def format_output(severity_idx: int, confidence: float) -> Dict[str, Any]:
    """
    Format inference output JSON with prediction label, confidence, severity, risk.
    """
    label = get_class_label(severity_idx)
    risk = _risk_level_from_severity(severity_idx)
    return {
        "prediction": label,
        "confidence": float(round(confidence, 4)),
        "severity_level": int(severity_idx),
        "risk_level": risk,
    }


class DRPredictor:
    """
    Loads the trained EfficientNet model and runs forward passes on single images.
    """

    def __init__(self, config: Optional[PredictorConfig] = None) -> None:
        print("Initializing DR predictor...")
        self.config = config or PredictorConfig()
        self.model = None
        self.device = "cpu"
        self._load_model()
        print("DR predictor initialized successfully")

    def _unwrap_state_dict(self, state):
        if isinstance(state, dict):
            for key in ("best_model_state_dict", "model_state_dict"):
                nested = state.get(key)
                if isinstance(nested, dict):
                    return nested
        return state

    def _extract_primary_logits(self, outputs):
        if isinstance(outputs, dict):
            return outputs["logits"]
        return outputs

    def _load_model(self) -> None:
        torch, _ = _lazy_import_torch()

        if self.config.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device

        logger.info("Loading EfficientNet model for inference on device=%s", self.device)

        weights_path = self.config.model_path
        state_dict = None
        model_name = self.config.model_name
        if weights_path.exists():
            logger.info("Loading model weights from %s", str(weights_path))
            state = torch.load(str(weights_path), map_location=self.device)
            state_dict = self._unwrap_state_dict(state)
            model_name = infer_model_family_from_state_dict(state_dict)

        self.model = build_dr_model(
            model_name=model_name,
            num_classes=len(DR_CLASSES.idx_to_label),
            pretrained=False,
            device=self.device,
        )

        if state_dict is not None:
            self.model.load_state_dict(state_dict)
            logger.info("Model weights loaded successfully.")
        else:
            logger.warning(
                "Model weights not found at %s. Using randomly initialized model.",
                str(weights_path),
            )

        self.model.eval()

    def _prepare_image(self, image: ImageInput) -> np.ndarray:
        if isinstance(image, (str, Path)):
            p = Path(image)
            if not p.exists():
                raise FileNotFoundError(f"Image not found: {p}")
            img = Image.open(p).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            arr = np.asarray(image)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            img = Image.fromarray(arr.astype(np.uint8), mode="RGB")

        # Our preprocessing pipeline returns normalized float32 HWC
        logger.info("Preprocessing image for inference in predictor...")
        arr = preprocess_for_inference(img)
        return arr

    def predict(self, image: ImageInput) -> Dict[str, Any]:
        """
        Run full inference pipeline:
          image -> preprocess -> model -> prediction dict.
        """
        torch, F = _lazy_import_torch()

        if self.model is None:
            raise RuntimeError("Model has not been loaded.")

        arr = self._prepare_image(image)

        # HWC -> CHW -> batch
        arr_chw = arr.transpose(2, 0, 1)
        x = torch.from_numpy(arr_chw).unsqueeze(0)  # shape: [1, C, H, W]
        x = x.to(self.device, dtype=torch.float32)

        with torch.no_grad():
            outputs = self._extract_primary_logits(self.model(x))
            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()

        severity_idx = int(np.argmax(probs))
        confidence = float(probs[severity_idx])

        logger.info(
            "Inference complete. severity=%d label=%s confidence=%.4f",
            severity_idx,
            get_class_label(severity_idx),
            confidence,
        )

        return format_output(severity_idx, confidence)

