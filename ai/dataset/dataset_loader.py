from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

import pandas as pd
from PIL import Image, ImageFile
try:
    from torch.utils.data import Dataset  # type: ignore
except Exception as e:  # pragma: no cover
    # Keep Phase-1 usable even if torch isn't installed/functional yet.
    # Later phases (training/inference) will require a working torch install.
    Dataset = object  # type: ignore
    logger = logging.getLogger(__name__)
    logger.warning("PyTorch unavailable; using Dataset fallback. error=%s", str(e))


logger = logging.getLogger(__name__)

# Pillow can raise errors on slightly-truncated files; this allows best-effort load.
ImageFile.LOAD_TRUNCATED_IMAGES = True


_IMG_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


@dataclass(frozen=True)
class AptosSample:
    image_path: Path
    label: int


class AptosDataset(Dataset):
    """
    APTOS Diabetic Retinopathy dataset loader.

    Expects the Kaggle APTOS label CSV format:
      - `id_code` column (image id without extension)
      - `diagnosis` column (0..4)
    """

    def __init__(
        self,
        *,
        csv_path: Union[str, Path],
        images_dir: Union[str, Path],
        transform: Optional[Callable[[Image.Image], object]] = None,
        allowed_exts: Sequence[str] = _IMG_EXTS,
        strict_labels: bool = True,
        max_resample_attempts: int = 3,
    ) -> None:
        print("Loading dataset...")

        self.csv_path = Path(csv_path)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.allowed_exts = tuple(e.lower() for e in allowed_exts)
        self.strict_labels = strict_labels
        self.max_resample_attempts = max(1, int(max_resample_attempts))

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        df = pd.read_csv(self.csv_path)
        required_cols = {"id_code", "diagnosis"}
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(
                f"CSV missing required columns {sorted(missing)}; "
                f"found columns: {list(df.columns)}"
            )

        samples: list[AptosSample] = []
        missing_images = 0
        corrupt_images = 0

        for _, row in df.iterrows():
            image_id = str(row["id_code"])
            label_raw = row["diagnosis"]

            try:
                label = int(label_raw)
            except Exception as e:
                if strict_labels:
                    raise ValueError(f"Invalid label for id_code={image_id}: {label_raw}") from e
                logger.warning(
                    "Skipping sample with invalid label. id_code=%s label=%r error=%s",
                    image_id,
                    label_raw,
                    str(e),
                )
                continue
            
            # Validate label range
            if label < 0 or label > 4:
                if strict_labels:
                    raise ValueError(f"Label out of range (0..4) for id_code={image_id}: {label}")
                logger.warning("Skipping sample with out-of-range label. id_code=%s label=%d", image_id, label)
                continue
            
            # Find image file
            image_path = None
            for ext in self.allowed_exts:
                candidate = self.images_dir / f"{image_id}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is None:
                missing_images += 1
                continue
            # Try opening image to check for corruption
            try:
                img = Image.open(image_path)
                img.verify()  # PIL check for corruption
            except Exception:
                corrupt_images += 1
                continue
            samples.append(AptosSample(image_path=image_path, label=label))

        if missing_images > 0:
            logger.warning(f"Skipped {missing_images} missing images during dataset initialization.")
        if corrupt_images > 0:
            logger.warning(f"Skipped {corrupt_images} corrupt images during dataset initialization.")

        if not samples:
            raise RuntimeError(
                "No valid samples found. Check csv_path/images_dir and file extensions."
            )

        self.samples = samples
        self.missing_images = missing_images

        print("Dataset loaded successfully")
        logger.info(
            "Dataset loaded successfully. samples=%d missing_images=%d csv=%s images_dir=%s",
            len(self.samples),
            self.missing_images,
            str(self.csv_path),
            str(self.images_dir),
        )

    def _resolve_image_path(self, image_id: str) -> Optional[Path]:
        """
        Resolve an image path from an id (without extension), trying common extensions.
        """
        # Many APTOS exports store images as <id>.png; we try a small set of extensions.
        for ext in self.allowed_exts:
            p = self.images_dir / f"{image_id}{ext}"
            if p.exists():
                return p

        # If files have mixed/unknown extensions, fall back to a glob.
        matches = list(self.images_dir.glob(f"{image_id}.*"))
        for p in matches:
            if p.suffix.lower() in self.allowed_exts and p.exists():
                return p

        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        attempts = 0
        current_idx = idx

        while attempts < self.max_resample_attempts:
            sample = self.samples[current_idx]
            try:
                img = Image.open(sample.image_path).convert("RGB")
                label = sample.label

                if self.transform is not None:
                    img = self.transform(img)

                return img, label
            except Exception as e:
                logger.exception(
                    "Failed to load image. idx=%s path=%s error=%s",
                    current_idx,
                    str(sample.image_path),
                    str(e),
                )
                attempts += 1
                current_idx = random.randrange(0, len(self.samples))

        raise RuntimeError(
            f"Failed to load image after {self.max_resample_attempts} attempts. "
            f"Last idx={current_idx} path={self.samples[current_idx].image_path}"
        )

