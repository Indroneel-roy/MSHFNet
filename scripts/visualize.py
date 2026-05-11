
import argparse
import os

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

from mshfnet.model import HybridSegmentationModel
from mshfnet.utils import predict_with_tta

matplotlib.use('Agg')  # non-interactive backend — safe for servers


# ── Colour map (matches Synapse convention) ───────────────────────────────────
# 0=bg  1=aorta  2=gallbladder  3=spleen  4=left kidney
# 5=liver  6=right kidney  7=stomach  8=pancreas
CMAP_COLORS = [
    (0.00, 0.00, 0.00),   # 0  background  — black
    (0.90, 0.10, 0.10),   # 1  aorta       — red
    (0.20, 0.80, 0.20),   # 2  gallbladder — green
    (0.10, 0.40, 0.90),   # 3  spleen      — blue
    (0.90, 0.60, 0.10),   # 4  left kidney — orange
    (0.60, 0.20, 0.80),   # 5  liver       — purple
    (0.10, 0.80, 0.80),   # 6  right kidney— cyan
    (0.90, 0.90, 0.10),   # 7  stomach     — yellow
    (0.90, 0.40, 0.70),   # 8  pancreas    — pink
]
ORGAN_CMAP = matplotlib.colors.ListedColormap(CMAP_COLORS)

ORGAN_LABELS = [
    'Background', 'Aorta', 'Gallbladder', 'Spleen',
    'Left Kidney', 'Liver', 'Right Kidney', 'Stomach', 'Pancreas',
]


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert integer label mask to an RGB image."""
    h, w   = mask.shape
    rgb    = np.zeros((h, w, 3), dtype=np.float32)
    for c, color in enumerate(CMAP_COLORS):
        where = mask == c
        rgb[where] = color
    return rgb


def save_figure(img_raw, gt_mask, pred_mask, save_path: str):
    """Save a three-panel figure: CT | Ground Truth | Prediction."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(img_raw, cmap='gray')
    axes[0].set_title('CT Input', fontsize=13)
    axes[0].axis('off')

    axes[1].imshow(img_raw, cmap='gray')
    if gt_mask is not None:
        axes[1].imshow(mask_to_rgb(gt_mask), alpha=0.5)
    axes[1].set_title('Ground Truth', fontsize=13)
    axes[1].axis('off')

    axes[2].imshow(img_raw, cmap='gray')
    axes[2].imshow(mask_to_rgb(pred_mask), alpha=0.5)
    axes[2].set_title('MSHFNet Prediction', fontsize=13)
    axes[2].axis('off')

    # Colour legend
    patches = [
        matplotlib.patches.Patch(color=CMAP_COLORS[c], label=ORGAN_LABELS[c])
        for c in range(1, 9)
    ]
    fig.legend(handles=patches, loc='lower center', ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(cfg: dict, checkpoint: str, input_path: str, output: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    # ── Load model ───────────────────────────────────────────────────────────
    model = HybridSegmentationModel(
        shared_dim  = cfg['shared_dim'],
        num_classes = cfg['num_classes'],
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and 'model_state' in state:
        model.load_state_dict(state['model_state'])
    else:
        model.load_state_dict(state)
    model.eval()
    print('Model loaded.')

    tta_kwargs = dict(
        image_size = cfg.get('image_size', 224),
        clip_min   = cfg.get('clip_min', -125.0),
        clip_max   = cfg.get('clip_max',  275.0),
    )

    # ── .npz single slice ────────────────────────────────────────────────────
    if input_path.endswith('.npz'):
        data    = np.load(input_path)
        img_raw = data['image']
        gt      = data.get('label', None)
        pred    = predict_with_tta(model, img_raw, device, **tta_kwargs)

        # Resize pred/gt to raw image resolution for display
        h, w = img_raw.shape
        pred_disp = np.array(
            Image.fromarray(pred.astype(np.uint8)).resize((w, h), Image.NEAREST)
        )
        gt_disp = None
        if gt is not None:
            gt_disp = gt.astype(np.uint8)

        save_figure(img_raw, gt_disp, pred_disp, output)

    # ── .h5 full volume ───────────────────────────────────────────────────────
    elif input_path.endswith('.h5'):
        h5f   = h5py.File(input_path, 'r')
        image = h5f['image'][:]   # (D, H, W)
        label = h5f.get('label')
        gt    = label[:] if label is not None else None
        h5f.close()

        os.makedirs(output, exist_ok=True)
        n_slices = image.shape[0]
        print(f'Visualising {n_slices} slices from {input_path}')

        for i in range(n_slices):
            img_raw = image[i]
            pred    = predict_with_tta(model, img_raw, device, **tta_kwargs)

            h, w = img_raw.shape
            pred_disp = np.array(
                Image.fromarray(pred.astype(np.uint8)).resize((w, h), Image.NEAREST)
            )
            gt_disp = gt[i].astype(np.uint8) if gt is not None else None

            slice_path = os.path.join(output, f'slice_{i:04d}.png')
            save_figure(img_raw, gt_disp, pred_disp, slice_path)

    else:
        raise ValueError(f'Unsupported input format: {input_path}. '
                         f'Expected .npz (single slice) or .h5 (volume).')


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualise MSHFNet predictions.')
    parser.add_argument('--config',     required=True,
                        help='Path to YAML config (e.g. configs/synapse.yaml)')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to model weights (.pth)')
    parser.add_argument('--input',      required=True,
                        help='.npz slice or .h5 volume to predict')
    parser.add_argument('--output',     required=True,
                        help='Output image path (.png) or directory (for .h5 volumes)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    main(cfg, checkpoint=args.checkpoint, input_path=args.input, output=args.output)
