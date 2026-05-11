#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MSHFNet evaluation script — 8-view TTA on Synapse test set.

Usage
-----
  python scripts/test.py \
      --config     configs/synapse.yaml \
      --checkpoint checkpoints/best_model.pth

Results are printed to stdout and optionally saved to a JSON file
with --output results.json.
"""

import argparse
import json
import os

import torch
import yaml

from mshfnet.model import HybridSegmentationModel
from mshfnet.utils import evaluate, print_results


# ─────────────────────────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(cfg: dict, checkpoint: str, output: str = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    # ── Load model ───────────────────────────────────────────────────────────
    print(f'Loading checkpoint: {checkpoint}')
    model = HybridSegmentationModel(
        shared_dim  = cfg['shared_dim'],
        num_classes = cfg['num_classes'],
    ).to(device)

    state = torch.load(checkpoint, map_location=device)
    # Support both raw state-dict and full checkpoint dict
    if isinstance(state, dict) and 'model_state' in state:
        model.load_state_dict(state['model_state'])
    else:
        model.load_state_dict(state)

    model.eval()
    print('Model loaded successfully.')

    # ── Evaluate ─────────────────────────────────────────────────────────────
    all_dice, all_hd95 = evaluate(
        model      = model,
        test_path  = cfg['test_path'],
        device     = device,
        image_size = cfg.get('image_size', 224),
        clip_min   = cfg.get('clip_min', -125.0),
        clip_max   = cfg.get('clip_max',  275.0),
    )

    mean_dsc, mean_hd = print_results(all_dice, all_hd95)

    # ── Optionally save results to JSON ──────────────────────────────────────
    if output:
        results = {
            'mean_dsc_pct' : round(mean_dsc, 4),
            'mean_hd95_mm' : round(mean_hd,  4),
            'per_organ'    : {
                organ: {
                    'dsc_pct': round(float(sum(v) / len(v)) * 100, 4) if v else None,
                    'hd95_mm': round(float(sum(all_hd95[organ]) / len(all_hd95[organ])), 4)
                               if all_hd95[organ] else None,
                }
                for organ, v in all_dice.items()
            },
        }
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nResults saved to: {output}')


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate MSHFNet with 8-view TTA.')
    parser.add_argument('--config',     required=True,
                        help='Path to YAML config (e.g. configs/synapse.yaml)')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to model weights (.pth)')
    parser.add_argument('--output',     default=None,
                        help='Optional path to save results as JSON')
    args = parser.parse_args()

    cfg = load_config(args.config)
    main(cfg, checkpoint=args.checkpoint, output=args.output)
