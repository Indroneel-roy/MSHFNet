
import argparse
import os
import time

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from mshfnet.dataset import SynapseDataset
from mshfnet.loss    import CLASS_WEIGHTS, DeepSupervisionLoss
from mshfnet.model   import HybridSegmentationModel


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device: torch.device) -> HybridSegmentationModel:
    model = HybridSegmentationModel(
        shared_dim  = cfg['shared_dim'],
        num_classes = cfg['num_classes'],
    ).to(device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Total params     : {total:,}')
    print(f'  Trainable params : {trainable:,}')
    return model


def build_optimizer(model: HybridSegmentationModel, cfg: dict) -> AdamW:
    """AdamW with layer-wise learning rates.

    Encoders (pretrained) use 0.1× base LR to avoid destroying pretrained weights.
    Fusion modules and decoder use full base LR.
    """
    return AdamW([
        {'params': model.cnn_encoder.parameters(),  'lr': cfg['encoder_lr']},
        {'params': model.swin_encoder.parameters(), 'lr': cfg['encoder_lr']},
        {'params': (
            list(model.fusion_s1.parameters()) +
            list(model.fusion_s2.parameters()) +
            list(model.fusion_s3.parameters()) +
            list(model.fusion_s4.parameters())
        ), 'lr': cfg['lr']},
        {'params': model.decoder.parameters(), 'lr': cfg['lr']},
    ], weight_decay=cfg['weight_decay'])


def build_criterion(cfg: dict, device: torch.device) -> DeepSupervisionLoss:
    class_weights = torch.tensor(cfg['class_weights'], dtype=torch.float32).to(device)
    return DeepSupervisionLoss(
        num_classes   = cfg['num_classes'],
        weights       = cfg['ds_weights'],
        class_weights = class_weights,
    ).to(device)


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_loss):
    torch.save({
        'epoch'           : epoch,
        'model_state'     : model.state_dict(),
        'optimizer_state' : optimizer.state_dict(),
        'scheduler_state' : scheduler.state_dict(),
        'best_loss'       : best_loss,
    }, path)
    print(f'  Checkpoint saved → {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Train one epoch
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    total_main = 0.0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs         = model(images)
        loss, main, _   = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_main += main.item()

        if batch_idx % 20 == 0:
            print(f'  Batch [{batch_idx:04d}/{len(loader)}] '
                  f'Loss: {loss.item():.4f}  Main: {main.item():.4f}')

    n = len(loader)
    return total_loss / n, total_main / n


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(cfg: dict, resume: str = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    if torch.cuda.is_available():
        print(f'GPU    : {torch.cuda.get_device_name(0)}')

    os.makedirs(cfg['save_path'], exist_ok=True)

    # ── Dataset & DataLoader ─────────────────────────────────────────────────
    all_files = sorted(
        f for f in os.listdir(cfg['train_path']) if f.endswith('.npz')
    )
    print(f'Training slices : {len(all_files)}')

    train_dataset = SynapseDataset(
        data_path  = cfg['train_path'],
        file_list  = all_files,
        mode       = 'train',
        image_size = cfg['image_size'],
        clip_min   = cfg['clip_min'],
        clip_max   = cfg['clip_max'],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size  = cfg['batch_size'],
        shuffle     = True,
        num_workers = cfg.get('num_workers', 4),
        pin_memory  = True,
    )

    # ── Model / Optimiser / Scheduler / Loss ─────────────────────────────────
    print('Building model...')
    model     = build_model(cfg, device)
    optimizer = build_optimizer(model, cfg)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg['epochs'], eta_min=cfg['min_lr']
    )
    criterion = build_criterion(cfg, device)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    best_loss   = float('inf')

    if resume:
        print(f'Resuming from {resume}')
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        best_loss   = ckpt['best_loss']
        print(f'  Resumed at epoch {start_epoch}  best_loss={best_loss:.4f}')

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {'train_loss': [], 'main_loss': []}

    print('Starting training...')
    print(f'  Model    : CNN (ResNet-50) + Swin-B')
    print(f'  Epochs   : {cfg["epochs"]}  (start={start_epoch})')
    print(f'  Batch    : {cfg["batch_size"]}')
    print(f'  LR       : {cfg["lr"]}  (encoders {cfg["encoder_lr"]})')
    print(f'  DS-w     : {cfg["ds_weights"]}')
    print('=' * 65)

    for epoch in range(start_epoch, cfg['epochs']):
        t0 = time.time()

        train_loss, main_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            grad_clip=cfg.get('grad_clip', 1.0),
        )
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['main_loss'].append(main_loss)

        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            best_path = os.path.join(cfg['save_path'], 'best_model.pth')
            torch.save(model.state_dict(), best_path)
            print(f'  ✓ Best model saved  (loss={best_loss:.4f})')

        # Periodic full checkpoint
        if (epoch + 1) % cfg.get('save_every', 10) == 0:
            ckpt_path = os.path.join(
                cfg['save_path'], f'checkpoint_epoch{epoch + 1}.pth'
            )
            save_checkpoint(ckpt_path, epoch, model, optimizer, scheduler, best_loss)

        elapsed = time.time() - t0
        print(f'Epoch [{epoch + 1:02d}/{cfg["epochs"]}]  '
              f'{elapsed:.1f}s  |  Loss: {train_loss:.4f}  |  Main: {main_loss:.4f}')

    print('=' * 65)
    print('Training complete!')
    print(f'Best training loss : {best_loss:.4f}')
    print(f'Best model saved to: {os.path.join(cfg["save_path"], "best_model.pth")}')


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MSHFNet on Synapse dataset.')
    parser.add_argument('--config', required=True,
                        help='Path to YAML config file (e.g. configs/synapse.yaml)')
    parser.add_argument('--resume', default=None,
                        help='Path to checkpoint .pth file to resume from')
    args = parser.parse_args()

    cfg = load_config(args.config)
    main(cfg, resume=args.resume)
