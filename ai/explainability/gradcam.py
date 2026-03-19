from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _lazy_import_torch():
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        logger.error("PyTorch is required for Grad-CAM: %s", str(e))
        raise RuntimeError(
            "PyTorch is required to compute Grad-CAM. "
            "Please ensure it is installed and configured correctly."
        ) from e
    return torch


@dataclass
class GradCAMConfig:
    """
    Grad-CAM configuration for EfficientNet-B0.
    """

    # Target layer name relative to model; for torchvision EfficientNet-B0,
    # the last conv features live under model.features[-1].
    target_layer: str = "features.7.1"  # robust default for many torchvision versions
    normalize_heatmap: bool = True


class GradCAM:
    """
    Generic Grad-CAM for a classifier model.

    Designed to work with the EfficientNet-B0 model we build in ai.model.efficientnet.
    """

    def __init__(self, model, config: Optional[GradCAMConfig] = None) -> None:
        print("Initializing Grad-CAM...")
        self.model = model
        self.config = config or GradCAMConfig()
        self.model.eval()

        self._features = None
        self._gradients = None

        self._register_hooks()
        print("Grad-CAM initialized successfully")

    def _get_target_module(self):
        """
        Resolve target module by dotted path from config.target_layer.
        """
        module = self.model
        for attr in self.config.target_layer.split("."):
            if not hasattr(module, attr):
                raise AttributeError(
                    f"Model has no attribute '{attr}' along target layer path "
                    f"'{self.config.target_layer}'"
                )
            module = getattr(module, attr)
        return module

    def _register_hooks(self) -> None:
        torch = _lazy_import_torch()

        target_module = self._get_target_module()

        def forward_hook(module, input, output):  # noqa: D401
            self._features = output.detach()

        def backward_hook(module, grad_in, grad_out):  # noqa: D401
            self._gradients = grad_out[0].detach()

        target_module.register_forward_hook(forward_hook)
        target_module.register_full_backward_hook(backward_hook)  # type: ignore[attr-defined]

        logger.info("Registered Grad-CAM hooks on layer: %s", self.config.target_layer)

    def generate(
        self,
        input_tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a single image tensor.

        Args:
            input_tensor: torch.Tensor of shape [1, C, H, W] on correct device.
            target_class: Optional class index. If None, uses argmax.

        Returns:
            heatmap: numpy array of shape [H, W] in [0, 1].
        """
        torch = _lazy_import_torch()

        self.model.zero_grad()

        outputs = self.model(input_tensor)
        if target_class is None:
            target_class = int(outputs.argmax(dim=1).item())

        logger.info("Generating Grad-CAM for target_class=%d", target_class)

        one_hot = torch.zeros_like(outputs)
        one_hot[0, target_class] = 1.0
        outputs.backward(gradient=one_hot, retain_graph=True)

        if self._features is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture features/gradients.")

        gradients = self._gradients  # [B, C, H', W']
        activations = self._features  # [B, C, H', W']

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (weights * activations).sum(dim=1, keepdim=False)  # [B, H', W']

        cam = cam[0]
        cam = torch.relu(cam)

        cam_np = cam.cpu().numpy()

        # Resize CAM to input spatial size
        _, _, H, W = input_tensor.shape
        cam_np = _resize_heatmap(cam_np, (H, W))

        if self.config.normalize_heatmap:
            cam_np = _normalize(cam_np)

        return cam_np


def _resize_heatmap(cam: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    import cv2  # type: ignore

    H, W = size
    cam_resized = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
    return cam_resized


def _normalize(cam: np.ndarray) -> np.ndarray:
    cam = cam.astype(np.float32)
    cam -= cam.min()
    max_val = cam.max()
    if max_val > 0:
        cam /= max_val
    return cam

