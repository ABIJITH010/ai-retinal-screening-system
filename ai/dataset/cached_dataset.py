# -*- coding: utf-8 -*-
"""
CachedAptosDataset
==================
Wraps AptosDataset with a disk-based pre-computation cache.

WHY THIS EXISTS
---------------
The full preprocessing pipeline (circular_crop → CLAHE → medianBlur → resize →
normalize) takes ~5-10 ms PER IMAGE on CPU.  With batch_size=64 and num_workers=0
that means ~320-640 ms of CPU work per batch while the GPU is completely idle.

WHAT THIS DOES
--------------
Pre-computes every sample ONCE and saves:
    dataset/cache/<id_code>.npy   ← float32 shape (3, 224, 224), CHW, ready for GPU

During training each __getitem__ call:
    - loads a 150 KB .npy file  (~0.5 ms, memory-mapped)
    - applies stochastic augmentation ONLY (random flip / colour jitter)
    - returns a tensor — no CLAHE, no medianBlur, no resize

RESULT
------
~10-20× faster data loading → GPU stays busy continuously.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


try:
    from torch.utils.data import Dataset  # type: ignore
    import torch
except Exception:
    Dataset = object  # type: ignore


class CachedAptosDataset(Dataset):
    """
    Loads pre-computed float32 CHW tensors from .npy cache files.
    Optionally applies a lightweight augmentation transform (flip, colour jitter)
    that is fast because it operates on already-processed tensors.

    Parameters
    ----------
    cache_dir   : directory containing <id_code>.npy files and labels.npy
    augment_fn  : optional callable(tensor) -> tensor for training augmentation
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        augment_fn: Optional[Callable] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.augment_fn = augment_fn

        labels_file = self.cache_dir / "labels.npy"
        ids_file = self.cache_dir / "ids.npy"

        if not labels_file.exists() or not ids_file.exists():
            raise FileNotFoundError(
                f"Cache not found at {self.cache_dir}. "
                "Run `python -m ai.dataset.build_cache` first."
            )

        self.labels: np.ndarray = np.load(str(labels_file))
        self.ids: list[str] = np.load(str(ids_file), allow_pickle=True).tolist()

        print(f"  [Cache] Loaded {len(self.ids)} pre-processed samples from {self.cache_dir}")
        logger.info("CachedAptosDataset ready. samples=%d cache=%s", len(self.ids), self.cache_dir)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        npy_path = self.cache_dir / f"{self.ids[idx]}.npy"
        # memory_map: avoids loading the whole file into RAM; OS page-cache does the rest
        arr = np.load(str(npy_path), mmap_mode="r").astype(np.float32)
        tensor = torch.from_numpy(arr.copy())          # CHW float32

        if self.augment_fn is not None:
            tensor = self.augment_fn(tensor)

        label = int(self.labels[idx])
        return tensor, label
