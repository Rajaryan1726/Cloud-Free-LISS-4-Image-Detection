
"""
Stage 2: Fine-tune the Stage-1 (SEN12MS-CR-pretrained) PConvUNet on
LISS-IV synthetic (cloudy, clean) pairs.

Key differences from Stage 1 training:
  - Model input here is only 3 channels (LISS-IV Green/Red/NIR cloudy),
    NOT 5 channels (no Sentinel-1 SAR available for LISS-IV) -- so the
    first conv layer's input channels must be adapted from the
    pretrained 5-channel checkpoint down to 3 channels.
  - Much smaller learning rate (fine-tuning, not training from scratch).
  - Fewer epochs typically needed.

Usage:
    python -m src.train_liss4 --epochs 15 --lr 1e-5 --resume checkpoints/best_model.pth
"""
import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .preprocessing import config as cfg
from .preprocessing.liss4_dataset import LISS4SyntheticDataset
from .models.pconv_unet import PConvUNet, count_parameters
from .losses import CombinedLoss


def _load_synthetic_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Synthetic manifest not found at {path}. Run "
            f"'python -m src.preprocessing.synthetic_clouds' first."
        )
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def adapt_first_conv_for_3ch(model: PConvUNet, pretrained_state_dict: dict):
    """
    The Stage-1 model's first conv layer expects 5 input channels
    (2 S1 + 3 S2-mapped). For LISS-IV fine-tuning we only have 3 channels
    (Green/Red/NIR, no SAR). We load all matching weights as-is, and for
    the first conv layer we take only the 3 channels that correspond to
    the S2-mapped bands (channels index 2,3,4 in the original 5-channel
    input order: S1(2ch) + S2cloudy_mapped(3ch)).
    """
    model_dict = model.state_dict()

    # Find the first conv layer's weight key (adjust name if your
    # PConvUNet names it differently -- check with model_dict.keys() if needed)
    first_conv_key = None
    for k in pretrained_state_dict.keys():
        if "weight" in k and pretrained_state_dict[k].dim() == 4:
            first_conv_key = k
            break

    if first_conv_key is None:
        raise RuntimeError("Could not find a first conv weight tensor in checkpoint")

    old_weight = pretrained_state_dict[first_conv_key]  # shape (out_ch, 5, k, k)

    if old_weight.shape[1] == 5:
        # keep only the last 3 input channels (the S2-mapped ones)
        new_weight = old_weight[:, 2:5, :, :].clone()
        pretrained_state_dict[first_conv_key] = new_weight
        print(f"  Adapted '{first_conv_key}': {old_weight.shape} -> {new_weight.shape}")
    elif old_weight.shape[1] == 3:
        print(f"  '{first_conv_key}' already 3-channel, no adaptation needed")
    else:
        raise RuntimeError(f"Unexpected input channel count: {old_weight.shape}")

    # load everything (strict=False to tolerate the first-layer shape we just fixed,
    # and any minor naming mismatches)
    model.load_state_dict(pretrained_state_dict, strict=False)
    return model


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            x = batch["cloudy"].to(device, non_blocking=True)   # (B,3,H,W)
            y = batch["clean"].to(device, non_blocking=True)    # (B,3,H,W)

            pred = model(x)
            loss, parts = criterion(pred, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += parts["total"]
            n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-5,
                         help="Fine-tuning LR -- much smaller than Stage 1's 1e-4")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--sam_weight", type=float, default=0.3)
    parser.add_argument("--resume", required=True,
                         help="Path to Stage-1 checkpoint (best_model.pth)")
    parser.add_argument("--checkpoint_dir", default=str(cfg.PROJECT_ROOT / "checkpoints_liss4"))
    parser.add_argument("--val_split", type=float, default=0.15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.batch_size is None:
        batch_size = cfg.CLOUD_BATCH_SIZE if device.type == "cuda" else cfg.LOCAL_BATCH_SIZE
    else:
        batch_size = args.batch_size
    print(f"Batch size: {batch_size}")

    manifest_rows = _load_synthetic_manifest(cfg.LISS4_MANIFEST_DIR / "liss4_synthetic_manifest.csv")
    full_ds = LISS4SyntheticDataset(manifest_rows)

    val_size = int(len(full_ds) * args.val_split)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train patches: {len(train_ds)}  Val patches: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    # Build model with 3-channel input (no SAR for LISS-IV)
    model = PConvUNet(in_channels=3, out_channels=3).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    print(f"Loading Stage-1 checkpoint from {args.resume}")
    ckpt = torch.load(args.resume, map_location=device)
    pretrained_state_dict = ckpt["model_state_dict"]
    model = adapt_first_conv_for_3ch(model, pretrained_state_dict)

    print(f"SAM loss weight: {args.sam_weight}")
    criterion = CombinedLoss(l1_weight=1.0, ssim_weight=1.0, sam_weight=args.sam_weight)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = checkpoint_dir / "best_model_liss4.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "sam_weight": args.sam_weight,
                "in_channels": 3,
            }, ckpt_path)
            print(f"  -> New best model saved to {ckpt_path} (val_loss={val_loss:.4f})")

    print(f"Fine-tuning complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()