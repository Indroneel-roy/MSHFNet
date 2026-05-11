# -*- coding: utf-8 -*-
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# Label index → organ name mapping (Synapse convention)
ORGAN_NAMES = [
    'Background',   # 0
    'Aorta',        # 1
    'Gallbladder',  # 2
    'Spleen',       # 3
    'Left Kidney',  # 4
    'Liver',        # 5
    'Right Kidney', # 6
    'Stomach',      # 7
    'Pancreas',     # 8
]


class SynapseDataset(Dataset):
    """PyTorch Dataset for the Synapse multi-organ segmentation benchmark.

    Parameters
    ----------
    data_path : str       — path to the directory containing .npz slice files.
    file_list : list[str] — list of .npz filenames to load.
    mode      : str       — 'train' (applies augmentation) or 'val'/'test'.
    image_size: int       — spatial size to resize slices to (default 224).
    clip_min  : float     — lower HU clip value (default -125).
    clip_max  : float     — upper HU clip value (default  275).
    """

    def __init__(
        self,
        data_path:  str,
        file_list:  list,
        mode:       str   = 'train',
        image_size: int   = 224,
        clip_min:   float = -125.0,
        clip_max:   float =  275.0,
    ):
        self.data_path  = data_path
        self.files      = file_list
        self.mode       = mode
        self.image_size = image_size
        self.clip_min   = clip_min
        self.clip_max   = clip_max

        print(f'[SynapseDataset] mode={mode}  slices={len(self.files)}')

    def __len__(self) -> int:
        return len(self.files)

    # ── Augmentation ──────────────────────────────────────────────────────────
    def augment(self, image: np.ndarray, label: np.ndarray):
        """Apply random spatial and intensity augmentations.

        Augmentations applied independently with p = 0.5 each:
          - Horizontal flip
          - Vertical flip
          - Random 90°/180°/270° rotation
          - Brightness scaling  [0.7, 1.3]
          - Gaussian noise      σ = 0.1
          - Contrast jitter     [0.8, 1.2]
          - Random crop + resize (zoom = [0.8, 1.0])
        """
        # Horizontal flip
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            label = np.fliplr(label).copy()

        # Vertical flip
        if random.random() > 0.5:
            image = np.flipud(image).copy()
            label = np.flipud(label).copy()

        # Random 90°/180°/270° rotation
        if random.random() > 0.5:
            k     = random.choice([1, 2, 3])
            image = np.rot90(image, k).copy()
            label = np.rot90(label, k).copy()

        # Brightness scaling
        if random.random() > 0.5:
            image = image * random.uniform(0.7, 1.3)

        # Gaussian noise
        if random.random() > 0.5:
            image = image + np.random.normal(0, 0.1, image.shape)

        # Contrast jitter
        if random.random() > 0.5:
            mean  = image.mean()
            image = (image - mean) * random.uniform(0.8, 1.2) + mean

        # Random crop + resize
        if random.random() > 0.5:
            h, w    = image.shape
            zoom    = random.uniform(0.8, 1.0)
            new_h   = int(h * zoom)
            new_w   = int(w * zoom)
            top     = random.randint(0, h - new_h)
            left    = random.randint(0, w - new_w)
            image   = image[top:top + new_h, left:left + new_w]
            label   = label[top:top + new_h, left:left + new_w]
            image   = np.array(Image.fromarray(image).resize((h, w)))
            label   = np.array(
                Image.fromarray(label.astype(np.uint8)).resize((h, w), Image.NEAREST)
            )

        return image, label

    # ── Preprocessing ─────────────────────────────────────────────────────────
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """HU clip → z-score normalisation → resize."""
        image = np.clip(image, self.clip_min, self.clip_max)
        image = (image - image.mean()) / (image.std() + 1e-8)
        image = np.array(
            Image.fromarray(image).resize((self.image_size, self.image_size))
        )
        return image

    # ── __getitem__ ───────────────────────────────────────────────────────────
    def __getitem__(self, idx: int):
        sample = np.load(os.path.join(self.data_path, self.files[idx]))
        image  = sample['image']   # (H, W) float
        label  = sample['label']   # (H, W) int

        # Resize label to target size (nearest-neighbour to preserve class indices)
        image = self.preprocess(image)
        label = np.array(
            Image.fromarray(label.astype(np.uint8)).resize(
                (self.image_size, self.image_size), Image.NEAREST
            )
        )

        if self.mode == 'train':
            image, label = self.augment(image, label)

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        label = torch.tensor(label, dtype=torch.long)                   # (H, W)
        return image, label
