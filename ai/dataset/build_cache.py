# -*- coding: utf-8 -*-
"""
build_cache.py
==============
One-time script: pre-processes every APTOS image through the full medical-grade
pipeline (circular_crop → CLAHE → medianBlur → resize → normalize) and saves the
result as a float32 CHW numpy array in dataset/cache/.

Run ONCE before training:
    python -m ai.dataset.build_cache

Output layout:
    dataset/cache/
        <id_code>.npy     ← float32 (3, 224, 224)
        labels.npy        ← int32  (N,)
        ids.npy           ← str    (N,)
        cache_info.txt    ← human-readable summary

After this, `python -m ai.training.train` will automatically detect the cache
and use CachedAptosDataset — giving 10-20× faster data loading.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_cache(
    csv_path: str = "dataset/labels.csv",
    images_dir: str = "dataset/images",
    cache_dir: str = "dataset/cache",
    target_size: tuple = (224, 224),
) -> None:
    import pandas as pd
    from ai.preprocessing.pipeline import preprocess_for_training

    csv_path_p = Path(csv_path)
    images_dir_p = Path(images_dir)
    cache_dir_p = Path(cache_dir)
    cache_dir_p.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  CACHE BUILDER — Pre-processing APTOS dataset")
    print("=" * 60)
    print(f"  CSV       : {csv_path_p}")
    print(f"  Images    : {images_dir_p}")
    print(f"  Cache dir : {cache_dir_p}")
    print(f"  Target    : {target_size[0]}x{target_size[1]} float32 CHW")
    print("=" * 60 + "\n")

    df = pd.read_csv(csv_path_p)
    IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    ids_out = []
    labels_out = []
    skipped = 0
    t0 = time.time()

    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        image_id = str(row["id_code"])
        label = int(row["diagnosis"])

        # Find image file
        image_path = None
        for ext in IMG_EXTS:
            candidate = images_dir_p / f"{image_id}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            logger.warning("Image not found, skipping: %s", image_id)
            skipped += 1
            continue

        cache_file = cache_dir_p / f"{image_id}.npy"

        # Skip if already cached (allows resuming interrupted runs)
        if cache_file.exists():
            ids_out.append(image_id)
            labels_out.append(label)
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                remaining = elapsed / (i + 1) * (total - i - 1)
                print(f"  [{i+1:>4}/{total}]  {image_id}  "
                      f"(skipped—cached)  ETA: {remaining:.0f}s")
            continue

        try:
            img = Image.open(image_path).convert("RGB")
            # Full pipeline: circular_crop → CLAHE → medianBlur → resize → normalize
            arr = preprocess_for_training(img)           # float32 HWC (224,224,3)

            # Convert HWC → CHW for direct use by PyTorch
            if arr.shape == (224, 224, 3):
                arr = arr.transpose(2, 0, 1)             # → (3, 224, 224)
            assert arr.shape == (3, 224, 224), f"Unexpected shape: {arr.shape}"
            assert arr.dtype == np.float32

            np.save(str(cache_file), arr)
            ids_out.append(image_id)
            labels_out.append(label)

            if (i + 1) % 100 == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                done = i + 1 - skipped
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (total - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1:>4}/{total}]  {image_id}  "
                      f"({rate:.1f} img/s)  ETA: {remaining:.0f}s")

        except Exception as e:
            logger.warning("Failed to preprocess %s: %s", image_id, e)
            skipped += 1
            continue

    # Save metadata arrays
    np.save(str(cache_dir_p / "ids.npy"), np.array(ids_out, dtype=object))
    np.save(str(cache_dir_p / "labels.npy"), np.array(labels_out, dtype=np.int32))

    elapsed_total = time.time() - t0
    info_lines = [
        f"Cached samples : {len(ids_out)}",
        f"Skipped        : {skipped}",
        f"Time taken     : {elapsed_total:.1f}s",
        f"Rate           : {len(ids_out)/elapsed_total:.1f} img/s",
        f"Cache dir      : {cache_dir_p.resolve()}",
        f"Array shape    : (3, 224, 224) float32",
    ]
    (cache_dir_p / "cache_info.txt").write_text("\n".join(info_lines))

    print("\n" + "=" * 60)
    print(f"  CACHE BUILD COMPLETE")
    print(f"  Cached : {len(ids_out)} images")
    print(f"  Skipped: {skipped}")
    print(f"  Time   : {elapsed_total:.1f}s  ({len(ids_out)/elapsed_total:.1f} img/s)")
    print(f"  Dir    : {cache_dir_p.resolve()}")
    print("=" * 60 + "\n")
    print("  You can now run training — it will use the cache automatically.")
    print("  Command: python -m ai.training.train\n")


if __name__ == "__main__":
    build_cache()
