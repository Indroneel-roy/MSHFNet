# -*- coding: utf-8 -*
import os

import h5py
import numpy as np
import torch
from PIL import Image
from medpy.metric.binary import dc, hd95


ORGAN_NAMES = [
    'Aorta',        # class 1
    'Gallbladder',  # class 2
    'Spleen',       # class 3
    'Left Kidney',  # class 4
    'Liver',        # class 5
    'Right Kidney', # class 6
    'Stomach',      # class 7
    'Pancreas',     # class 8
]


# ─────────────────────────────────────────────────────────────────────────────
# 8-View Test-Time Augmentation 
# ─────────────────────────────────────────────────────────────────────────────
def predict_with_tta(
    model:     torch.nn.Module,
    slice_2d:  np.ndarray,
    device:    torch.device,
    image_size: int = 224,
    clip_min:   float = -125.0,
    clip_max:   float =  275.0,
) -> np.ndarray:


    def preprocess(s: np.ndarray) -> torch.Tensor:
        s = np.clip(s, clip_min, clip_max)
        s = (s - s.mean()) / (s.std() + 1e-8)
        s = np.array(Image.fromarray(s).resize((image_size, image_size)))
        return torch.tensor(s, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    def get_pred(inp: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = model(inp)
        return torch.softmax(out, dim=1).squeeze(0)   # (num_classes, H, W)

    preds = []
    for k in range(4):                                # 0°, 90°, 180°, 270°
        # Original orientation for this rotation
        rotated = np.rot90(slice_2d, k).copy()
        p = get_pred(preprocess(rotated))
        p = torch.rot90(p, -k, dims=[1, 2])           # undo rotation
        preds.append(p)

        # Horizontally flipped
        flipped = np.fliplr(rotated).copy()
        p2 = get_pred(preprocess(flipped))
        p2 = torch.flip(p2, dims=[2])                 # undo flip
        p2 = torch.rot90(p2, -k, dims=[1, 2])         # undo rotation
        preds.append(p2)

    # Average all 8 predictions in probability space
    avg_pred = sum(preds) / len(preds)
    return torch.argmax(avg_pred, dim=0).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Full-volume evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(
    model:      torch.nn.Module,
    test_path:  str,
    device:     torch.device,
    image_size: int   = 224,
    clip_min:   float = -125.0,
    clip_max:   float =  275.0,
) -> tuple:

    model.eval()

    all_dice = {organ: [] for organ in ORGAN_NAMES}
    all_hd95 = {organ: [] for organ in ORGAN_NAMES}

    test_files = sorted(os.listdir(test_path))
    print(f'Evaluating {len(test_files)} test volumes...')
    print('=' * 55)

    for fname in test_files:
        print(f'  Processing: {fname}')
        h5f   = h5py.File(os.path.join(test_path, fname), 'r')
        image = h5f['image'][:]   # (D, H, W)
        label = h5f['label'][:]   # (D, H, W)
        h5f.close()

        n_slices    = image.shape[0]
        pred_volume = np.zeros_like(label)

        for i in range(n_slices):
            pred = predict_with_tta(
                model, image[i], device,
                image_size=image_size,
                clip_min=clip_min,
                clip_max=clip_max,
            )
            # Resize prediction back to original label resolution
            pred = np.array(
                Image.fromarray(pred.astype(np.uint8)).resize(
                    (label.shape[2], label.shape[1]), Image.NEAREST
                )
            )
            pred_volume[i] = pred

        # Per-organ metrics
        for c, organ in enumerate(ORGAN_NAMES):
            organ_idx   = c + 1              # class indices start at 1
            pred_organ  = (pred_volume == organ_idx)
            label_organ = (label       == organ_idx)

            if label_organ.sum() > 0:
                dice = dc(pred_organ, label_organ)
                all_dice[organ].append(dice)
                if pred_organ.sum() > 0:
                    hd = hd95(pred_organ, label_organ)
                    all_hd95[organ].append(hd)
                else:
                    all_hd95[organ].append(100.0)   # penalise empty prediction

    return all_dice, all_hd95


# ─────────────────────────────────────────────────────────────────────────────
# Results table
# ─────────────────────────────────────────────────────────────────────────────
def print_results(all_dice: dict, all_hd95: dict) -> tuple:

    print('\n' + '=' * 55)
    print(f'{"Organ":<16} {"DSC (%)":>10} {"HD95 (mm)":>10}')
    print('=' * 55)

    mean_dices = []
    mean_hd95s = []

    for organ in ORGAN_NAMES:
        if all_dice[organ]:
            dice = np.mean(all_dice[organ]) * 100
            hd   = np.mean(all_hd95[organ])
            mean_dices.append(dice)
            mean_hd95s.append(hd)
            print(f'{organ:<16} {dice:>9.2f}% {hd:>10.2f}')
        else:
            print(f'{organ:<16} {"N/A":>10} {"N/A":>10}')

    mean_dsc = float(np.mean(mean_dices))
    mean_hd  = float(np.mean(mean_hd95s))

    print('=' * 55)
    print(f'{"Mean":<16} {mean_dsc:>9.2f}% {mean_hd:>10.2f}')
    print('=' * 55)

    return mean_dsc, mean_hd
