%%writefile /kaggle/working/Cloud-Free-LISS-4-Image-Detection/src/train.py
"""
Stage 1 training script: pretrain PConvUNet on SEN12MS-CR.
v2: increased SAM loss weight (spectral consistency) + LR scheduler.
"""
import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .preprocessing import config as cfg
from .preprocessing.dataset import SEN12MSCRDataset
from .models.pconv_unet import PConvUNet, count_parameters
from .losses import CombinedLoss

LISS4_BAND_IDX = list(cfg.S2_BAND_INDEX_FOR_LISS4.values())


def _load_split_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found at {path}. Run "
            f"'python -m src.preprocessing.split' first."
        )
    with open(path, newline="") as f:
        rows = []
        for r in csv.DictReader(f):
            r["has_clean"] = r["has_clean"] in ("True", "true", "1")
            rows.append(r)
        return rows


def make_model_inputs(batch: dict):
    s1 = batch["s1"]
    s2_cloudy_mapped = batch["s2_cloudy"][:, LISS4_BAND_IDX, :, :]
    x = torch.cat([s1, s2_cloudy_mapped], dim=1)
    y = batch["s2_clean"][:, LISS4_BAND_IDX, :, :]
    return x, y


def run_epoch(model, loader, criterion, optimizer, device, train: bool, max_batches: int = None):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            if max_batches is not None and n_batches >= max_batches:
                break

            x, y = make_model_inputs(batch)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

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
    parser.add_argument("--season", default="spring")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--checkpoint_dir", default=str(cfg.PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--sam_weight", type=float, default=0.3,
                         help="Higher = more emphasis on spectral consistency (color accuracy)")
    parser.add_argument("--resume", default=None,
                         help="Path to a checkpoint to resume model weights from")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.batch_size is None:
        batch_size = cfg.CLOUD_BATCH_SIZE if device.type == "cuda" else cfg.LOCAL_BATCH_SIZE
    else:
        batch_size = args.batch_size
    print(f"Batch size: {batch_size}")

    train_rows = _load_split_csv(cfg.SPLIT_DIR / f"{args.season}_train.csv")
    val_rows = _load_split_csv(cfg.SPLIT_DIR / f"{args.season}_val.csv")

    train_ds = SEN12MSCRDataset(train_rows, require_clean=True)
    val_ds = SEN12MSCRDataset(val_rows, require_clean=True)
    print(f"Train patches: {len(train_ds)}  Val patches: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = PConvUNet(in_channels=5, out_channels=3).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    start_epoch = 1
    if args.resume:
        print(f"Resuming weights from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    print(f"SAM loss weight: {args.sam_weight}")
    criterion = CombinedLoss(l1_weight=1.0, ssim_weight=1.0, sam_weight=args.sam_weight)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True, max_batches=args.max_batches)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False, max_batches=args.max_batches)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = checkpoint_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "sam_weight": args.sam_weight,
            }, ckpt_path)
            print(f"  -> New best model saved to {ckpt_path} (val_loss={val_loss:.4f})")

    print(f"Training complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()