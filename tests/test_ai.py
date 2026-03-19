import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from ai.model.efficientnet import infer_model_family_from_state_dict
from ai.inference.predictor import _risk_level_from_severity, format_output
from ai.preprocessing.image_cleaning import CleaningConfig
from ai.preprocessing.pipeline import PreprocessConfig, preprocess_for_inference
from ai.reliability.quality import QualityConfig, assess_image_quality
from ai.reliability.reliability import run_reliable_inference
from ai.training.trainer import DRTrainer
from database.schema import insert_screening


def test_format_output_maps_prediction_fields() -> None:
    result = format_output(2, 0.87654)

    assert result == {
        "prediction": "Moderate",
        "confidence": 0.8765,
        "severity_level": 2,
        "risk_level": "Medium",
    }


def test_risk_level_mapping_covers_expected_ranges() -> None:
    assert _risk_level_from_severity(0) == "Low"
    assert _risk_level_from_severity(2) == "Medium"
    assert _risk_level_from_severity(4) == "High"


def test_infer_model_family_detects_ensemble_weights() -> None:
    ensemble_state = {
        "efficientnet_head.1.weight": torch.randn(5, 1280),
        "fusion_head.1.weight": torch.randn(512, 1792),
    }
    single_state = {
        "features.0.0.weight": torch.randn(32, 3, 3, 3),
        "classifier.1.weight": torch.randn(5, 1280),
    }

    assert infer_model_family_from_state_dict(ensemble_state) == "ensemble"
    assert infer_model_family_from_state_dict(single_state) == "efficientnet_b0"


def test_preprocess_for_inference_returns_normalized_image() -> None:
    image = Image.fromarray(np.full((48, 64, 3), 128, dtype=np.uint8), mode="RGB")
    config = PreprocessConfig(
        cleaning=CleaningConfig(target_size=(32, 32), normalization="zero_one"),
    )

    processed = preprocess_for_inference(image, config=config)

    assert processed.shape == (32, 32, 3)
    assert processed.dtype == np.float32
    assert float(processed.min()) >= 0.0
    assert float(processed.max()) <= 1.0


def test_assess_image_quality_returns_expected_metrics() -> None:
    gradient = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    image = Image.fromarray(np.stack([gradient, gradient, gradient], axis=-1), mode="RGB")

    metrics = assess_image_quality(image)

    assert {
        "blur_score",
        "brightness",
        "contrast",
        "is_blurry",
        "is_too_dark",
        "is_too_bright",
        "is_low_contrast",
        "has_warnings",
        "quality_status",
        "warning_reasons",
        "hard_failure_reasons",
        "is_valid",
    }.issubset(metrics.keys())
    assert isinstance(metrics["is_valid"], bool)


def test_assess_image_quality_can_warn_without_rejecting() -> None:
    image = Image.fromarray(np.full((64, 64, 3), 127, dtype=np.uint8), mode="RGB")
    config = QualityConfig(
        blur_warning_threshold=1e6,
        blur_reject_threshold=-1.0,
        brightness_warning_range=(0.0, 1.0),
        brightness_reject_range=(0.0, 1.0),
        contrast_warning_threshold=-1.0,
        contrast_reject_threshold=-1.0,
    )

    metrics = assess_image_quality(image, config=config)

    assert metrics["is_valid"] is True
    assert metrics["has_warnings"] is True
    assert metrics["quality_status"] == "warning"
    assert "blurry" in metrics["warning_reasons"]


def test_assess_image_quality_rejects_severely_bad_images() -> None:
    image = Image.fromarray(np.full((64, 64, 3), 127, dtype=np.uint8), mode="RGB")
    config = QualityConfig(
        blur_warning_threshold=1e6,
        blur_reject_threshold=1e6,
        brightness_warning_range=(0.0, 1.0),
        brightness_reject_range=(0.0, 1.0),
        contrast_warning_threshold=-1.0,
        contrast_reject_threshold=-1.0,
    )

    metrics = assess_image_quality(image, config=config)

    assert metrics["is_valid"] is False
    assert metrics["quality_status"] == "reject"
    assert "extremely_blurry" in metrics["hard_failure_reasons"]


def test_insert_screening_persists_record(tmp_path) -> None:
    db_path = tmp_path / "retinal_screening.db"

    screening_id = insert_screening(
        prediction="Moderate",
        confidence=0.82,
        severity_level=2,
        risk_level="Medium",
        db_path=db_path,
    )

    assert screening_id == 1

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT prediction, confidence, severity_level, risk_level FROM screenings WHERE id = ?",
            (screening_id,),
        ).fetchone()

    assert row == ("Moderate", 0.82, 2, "Medium")


def test_reliable_inference_allows_warning_quality_images(monkeypatch) -> None:
    class _PredictorStub:
        def __init__(self, *_args, **_kwargs) -> None:
            self.device = "cpu"
            self.model = nn.Identity()

    monkeypatch.setattr(
        "ai.reliability.reliability.assess_image_quality",
        lambda *_args, **_kwargs: {
            "blur_score": 5.0,
            "brightness": 0.2,
            "contrast": 0.1,
            "is_blurry": True,
            "is_too_dark": False,
            "is_too_bright": False,
            "is_low_contrast": False,
            "has_warnings": True,
            "quality_status": "warning",
            "warning_reasons": ["blurry"],
            "hard_failure_reasons": [],
            "is_valid": True,
        },
    )
    monkeypatch.setattr("ai.reliability.reliability.DRPredictor", _PredictorStub)
    monkeypatch.setattr(
        "ai.reliability.reliability.mc_dropout_predict",
        lambda *_args, **_kwargs: {
            "mean_probs": np.array([0.05, 0.10, 0.75, 0.05, 0.05], dtype=np.float32),
            "std_probs": np.array([0.01, 0.01, 0.02, 0.01, 0.01], dtype=np.float32),
        },
    )
    monkeypatch.setattr(
        "ai.preprocessing.pipeline.preprocess_for_inference",
        lambda *_args, **_kwargs: np.zeros((224, 224, 3), dtype=np.float32),
    )

    image = Image.fromarray(np.full((64, 64, 3), 127, dtype=np.uint8), mode="RGB")
    result = run_reliable_inference(image)

    assert result["prediction"] == "Moderate"
    assert result["quality_status"] == "warning"
    assert result["quality_warning"] is True
    assert result["quality"]["warning_reasons"] == ["blurry"]


class _TinyEfficientNetLike(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(8, 5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class _FlipSensitiveModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left_edge = x[..., 0].mean(dim=(1, 2))
        right_edge = x[..., -1].mean(dim=(1, 2))
        return torch.stack(
            [
                left_edge,
                right_edge,
                left_edge - right_edge,
                right_edge - left_edge,
                left_edge + right_edge,
            ],
            dim=1,
        ) + self.dummy


class _TinyEnsembleLike(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.efficientnet = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.resnet = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.efficientnet_head = nn.Linear(4, 5)
        self.resnet_head = nn.Linear(4, 5)
        self.fusion_head = nn.Linear(8, 5)

    def forward(self, x: torch.Tensor):
        efficientnet_features = self.efficientnet(x)
        resnet_features = self.resnet(x)
        efficientnet_logits = self.efficientnet_head(efficientnet_features)
        resnet_logits = self.resnet_head(resnet_features)
        fusion_logits = self.fusion_head(torch.cat([efficientnet_features, resnet_features], dim=1))
        return {
            "logits": fusion_logits,
            "efficientnet_logits": efficientnet_logits,
            "resnet_logits": resnet_logits,
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


def _trainer_config(tmp_path):
    return SimpleNamespace(
        epochs=30,
        learning_rate=1e-4,
        batch_size=4,
        model_weights_path=tmp_path / "dr_model.pth",
        model_name="ensemble",
        use_class_weights=True,
        num_workers=0,
        pin_memory=False,
        val_split=0.2,
        shuffle_dataset=True,
        seed=42,
    )


def test_trainer_unfreezes_backbone_and_uses_lower_backbone_lr(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(DRTrainer, "_build_model", lambda self: _TinyEfficientNetLike())
    trainer = DRTrainer(_trainer_config(tmp_path))

    assert trainer.backbone_frozen is True
    assert all(not param.requires_grad for param in trainer.model.features.parameters())

    trainer._maybe_unfreeze_backbone(epoch=trainer.backbone_warmup_epochs + 1)

    backbone_group = next(group for group in trainer.optimizer.param_groups if group.get("group_name") == "backbone")
    head_group = next(group for group in trainer.optimizer.param_groups if group.get("group_name") == "head")

    assert trainer.backbone_frozen is False
    assert all(param.requires_grad for param in trainer.model.features.parameters())
    assert backbone_group["lr"] == pytest.approx(head_group["lr"] * trainer.unfreeze_lr_scale)


def test_trainer_unfreezes_all_ensemble_backbone_groups(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(DRTrainer, "_build_model", lambda self: _TinyEnsembleLike())
    trainer = DRTrainer(_trainer_config(tmp_path))

    assert trainer.backbone_frozen is True
    trainer._maybe_unfreeze_backbone(epoch=trainer.backbone_warmup_epochs + 1)

    backbone_groups = [
        group
        for group in trainer.optimizer.param_groups
        if "backbone" in str(group.get("group_name", ""))
    ]
    head_group = next(
        group
        for group in trainer.optimizer.param_groups
        if "backbone" not in str(group.get("group_name", ""))
    )

    assert trainer.backbone_frozen is False
    assert all(param.requires_grad for param in trainer.model.efficientnet.parameters())
    assert all(param.requires_grad for param in trainer.model.resnet.parameters())
    assert len(backbone_groups) == 2
    for backbone_group in backbone_groups:
        assert backbone_group["lr"] == pytest.approx(head_group["lr"] * trainer.unfreeze_lr_scale)


def test_trainer_eval_tta_averages_logits_with_horizontal_flip(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(DRTrainer, "_build_model", lambda self: _FlipSensitiveModel())
    trainer = DRTrainer(_trainer_config(tmp_path))

    image = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
    image[..., 0] = 2.0
    image[..., -1] = 1.0

    raw_logits = trainer.model(image)
    flipped_logits = trainer.model(torch.flip(image, dims=(-1,)))
    tta_logits = trainer._forward_logits(image, train=False)

    assert torch.allclose(tta_logits, 0.5 * (raw_logits + flipped_logits))


def test_trainer_best_model_checkpoint_saves_current_model_weights(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(DRTrainer, "_build_model", lambda self: _TinyEfficientNetLike())
    config = _trainer_config(tmp_path)
    trainer = DRTrainer(config)

    with torch.no_grad():
        for index, parameter in enumerate(trainer.model.parameters(), start=1):
            parameter.fill_(0.1 * index)

    expected_state = {
        key: value.detach().cpu().clone()
        for key, value in trainer.model.state_dict().items()
    }

    trainer._save_best_model(epoch=3)

    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.zero_()

    saved_state = torch.load(config.model_weights_path, map_location="cpu")

    assert saved_state.keys() == expected_state.keys()
    for key in expected_state:
        assert torch.equal(saved_state[key], expected_state[key])


def test_trainer_mixup_regularization_blends_images_and_targets(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(DRTrainer, "_build_model", lambda self: _TinyEfficientNetLike())
    trainer = DRTrainer(_trainer_config(tmp_path))
    trainer.cutmix_probability = 0.0
    trainer.mixup_probability = 1.0

    monkeypatch.setattr(np.random, "rand", lambda: 0.0)
    monkeypatch.setattr(np.random, "beta", lambda _a, _b: 0.7)
    monkeypatch.setattr(torch, "randperm", lambda n, device=None: torch.tensor([1, 0], device=device))

    images = torch.stack(
        [
            torch.ones((3, 4, 4), dtype=torch.float32),
            torch.zeros((3, 4, 4), dtype=torch.float32),
        ],
        dim=0,
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    mixed_images, mixed_targets = trainer._apply_batch_regularization(images, labels)

    assert mixed_targets is not None
    assert mixed_targets["lam"] == pytest.approx(0.7)
    assert torch.equal(mixed_targets["permuted_labels"], torch.tensor([1, 0]))
    assert torch.allclose(mixed_images[0], torch.full((3, 4, 4), 0.7))
