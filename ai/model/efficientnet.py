from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DRClassMapping:
    """Semantic mapping for DR severity levels."""

    idx_to_label: Dict[int, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:  # type: ignore[override]
        object.__setattr__(
            self,
            "idx_to_label",
            {
                0: "No DR",
                1: "Mild",
                2: "Moderate",
                3: "Severe",
                4: "Proliferative",
            },
        )


DR_CLASSES = DRClassMapping()


def _lazy_import_torch():
    """
    Lazily import torch/torchvision so the module can be imported
    even if these heavy deps are not installed/configured yet.
    """
    try:
        import torch  # type: ignore
        from torch import nn  # type: ignore
        from torchvision import models  # type: ignore
    except Exception as e:  # pragma: no cover
        logger.error("PyTorch / torchvision not available: %s", str(e))
        raise RuntimeError(
            "PyTorch and torchvision are required to build the EfficientNet model. "
            "Please ensure both are installed and working on this system."
        ) from e
    return torch, nn, models


def _build_raw_efficientnet_b0(models, *, pretrained: bool):
    # Handle API differences across torchvision versions
    try:
        if pretrained:
            try:
                weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1  # type: ignore[attr-defined]
            except AttributeError:  # pragma: no cover
                weights = "IMAGENET1K_V1"
            return models.efficientnet_b0(weights=weights)  # type: ignore[arg-type]
        return models.efficientnet_b0(weights=None)
    except TypeError:
        return models.efficientnet_b0(pretrained=pretrained)  # type: ignore[call-arg]


def _build_raw_resnet18(models, *, pretrained: bool):
    try:
        if pretrained:
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1  # type: ignore[attr-defined]
            except AttributeError:  # pragma: no cover
                weights = "IMAGENET1K_V1"
            return models.resnet18(weights=weights)  # type: ignore[arg-type]
        return models.resnet18(weights=None)
    except TypeError:
        return models.resnet18(pretrained=pretrained)  # type: ignore[call-arg]


def infer_model_family_from_state_dict(state_dict) -> str:
    if not isinstance(state_dict, dict):
        return "efficientnet_b0"

    keys = {str(key) for key in state_dict.keys()}
    ensemble_prefixes = (
        "efficientnet.",
        "resnet.",
        "efficientnet_head.",
        "resnet_head.",
        "fusion_head.",
    )
    if any(key.startswith(ensemble_prefixes) for key in keys):
        return "ensemble"
    return "efficientnet_b0"


def build_efficientnet_b0(
    *,
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.40,
    device: Optional[str] = None,
):
    """
    Build an EfficientNet-B0 model for 5-class DR classification.

    - Loads torchvision's EfficientNet-B0 (optionally with ImageNet weights)
    - Replaces the classifier head to output `num_classes`
    - Keeps everything CPU-compatible by default (device='cpu' if None)
    """
    logger.info(
        "Building EfficientNet-B0 model... num_classes=%d pretrained=%s dropout=%.3f device=%s",
        num_classes,
        pretrained,
        dropout,
        device or "cpu",
    )

    torch, nn, models = _lazy_import_torch()

    if device is None:
        device = "cpu"

    model = _build_raw_efficientnet_b0(models, pretrained=pretrained)

    # Replace classifier head
    in_features = model.classifier[1].in_features  # type: ignore[index]
    classifier = [
        nn.Dropout(p=float(dropout)),
        nn.Linear(in_features, num_classes),
    ]
    model.classifier = nn.Sequential(*classifier)  # type: ignore[assignment]

    model.to(device)
    logger.info("EfficientNet-B0 model built successfully on device=%s", device)
    return model


def build_dual_backbone_ensemble(
    *,
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.40,
    device: Optional[str] = None,
):
    """
    Build a lightweight dual-backbone ensemble for retinal grading.

    The model combines EfficientNet-B0 and ResNet18 feature extractors, then
    learns a small fusion head on top of both feature streams.
    """
    logger.info(
        "Building dual-backbone DR ensemble... num_classes=%d pretrained=%s dropout=%.3f device=%s",
        num_classes,
        pretrained,
        dropout,
        device or "cpu",
    )

    torch, nn, models = _lazy_import_torch()

    if device is None:
        device = "cpu"

    efficientnet = _build_raw_efficientnet_b0(models, pretrained=pretrained)
    efficientnet_feature_dim = efficientnet.classifier[1].in_features  # type: ignore[index]
    efficientnet.classifier = nn.Identity()  # type: ignore[assignment]

    resnet = _build_raw_resnet18(models, pretrained=pretrained)
    resnet_feature_dim = resnet.fc.in_features  # type: ignore[assignment]
    resnet.fc = nn.Identity()  # type: ignore[assignment]

    class DREnsembleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.efficientnet = efficientnet
            self.resnet = resnet
            self.efficientnet_head = nn.Sequential(
                nn.Dropout(p=float(dropout)),
                nn.Linear(efficientnet_feature_dim, num_classes),
            )
            self.resnet_head = nn.Sequential(
                nn.Dropout(p=float(dropout)),
                nn.Linear(resnet_feature_dim, num_classes),
            )
            self.fusion_head = nn.Sequential(
                nn.Dropout(p=float(dropout)),
                nn.Linear(efficientnet_feature_dim + resnet_feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=float(dropout)),
                nn.Linear(512, num_classes),
            )

        def forward(self, x):
            efficientnet_features = self.efficientnet(x)
            resnet_features = self.resnet(x)

            efficientnet_logits = self.efficientnet_head(efficientnet_features)
            resnet_logits = self.resnet_head(resnet_features)
            fusion_logits = self.fusion_head(
                torch.cat([efficientnet_features, resnet_features], dim=1)
            )
            logits = (
                0.50 * fusion_logits
                + 0.25 * efficientnet_logits
                + 0.25 * resnet_logits
            )
            return {
                "logits": logits,
                "efficientnet_logits": efficientnet_logits,
                "resnet_logits": resnet_logits,
                "fusion_logits": fusion_logits,
            }

        def get_backbone_parameter_groups(self):
            return {
                "efficientnet_backbone": list(self.efficientnet.parameters()),
                "resnet_backbone": list(self.resnet.parameters()),
            }

        def get_head_parameter_groups(self):
            return {
                "efficientnet_head": list(self.efficientnet_head.parameters()),
                "resnet_head": list(self.resnet_head.parameters()),
                "fusion_head": list(self.fusion_head.parameters()),
            }

        def set_backbone_trainable(self, trainable: bool) -> None:
            for parameter in self.efficientnet.parameters():
                parameter.requires_grad = trainable
            for parameter in self.resnet.parameters():
                parameter.requires_grad = trainable

        def set_backbone_eval(self) -> None:
            self.efficientnet.eval()
            self.resnet.eval()

    model = DREnsembleModel()
    model.to(device)
    logger.info("Dual-backbone DR ensemble built successfully on device=%s", device)
    return model


def build_dr_model(
    *,
    model_name: str = "ensemble",
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.40,
    device: Optional[str] = None,
):
    normalized_name = model_name.lower()
    if normalized_name in {"ensemble", "dual_backbone", "multi_neural"}:
        return build_dual_backbone_ensemble(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            device=device,
        )
    return build_efficientnet_b0(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        device=device,
    )


def get_class_label(index: int) -> str:
    """
    Map numeric class index to human-readable DR severity label.
    """
    try:
        return DR_CLASSES.idx_to_label[index]
    except KeyError as e:
        raise ValueError(f"Invalid DR class index: {index}") from e

