
"""
Evaluation script. NOTE: SEN12MSCRDataset already normalizes S2 data to
[0,1] internally (via _normalize_s2), so tensors coming out of the
DataLoader are already in [0,1] -- do NOT re-normalize them here.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as sk_ssim

from .preprocessing import config as cfg
from .preprocessing.dataset import SEN12MSCRDataset
from .models.pconv_unet import PConvUNet
from .train import make_model_inputs, _load_split_csv

LISS4_IDX = list(cfg.S2_BAND_INDEX_FOR_LISS4.values())


def compute_sam(pred, target, eps=1e-8):
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    dot = np.sum(p * t, axis=0)
    norm_p = np.linalg.norm(p, axis=0)
    norm_t = np.linalg.norm(t, axis=0)
    cos_angle = np.clip(dot / (norm_p * norm_t + eps), -1.0, 1.0)
    return float(np.degrees(np.mean(np.arccos(cos_angle))))


def compute_psnr_safe(pred01, target01, data_range=1.0, eps=1e-10):
    mse = max(np.mean((pred01 - target01) ** 2), eps)
    return 10 * np.log10((data_range ** 2) / mse)


def clip01(img):
    # data is ALREADY [0,1]-normalized by the Dataset class; just clip
    # defensively in case the model predicts slightly outside range.
    return np.clip(img, 0.0, 1.0)


def to_uint8_hwc(img01):
    return (np.transpose(img01, (1, 2, 0)) * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default="spring")
    parser.add_argument("--checkpoint", default=str(cfg.PROJECT_ROOT / "checkpoints" / "best_model.pth"))
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--out_dir", default=str(cfg.PROJECT_ROOT / "eval_results"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    batch_size = args.batch_size or (cfg.CLOUD_BATCH_SIZE if device.type == "cuda" else cfg.LOCAL_BATCH_SIZE)

    test_rows = _load_split_csv(cfg.SPLIT_DIR / f"{args.season}_test.csv")
    test_ds = SEN12MSCRDataset(test_rows, require_clean=True)
    print(f"Test patches: {len(test_ds)}")

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = PConvUNet(in_channels=5, out_channels=3).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} (val_loss={ckpt.get('val_loss', '?')})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_psnr, all_ssim, all_sam, input_mse = [], [], [], []
    saved_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            x, y = make_model_inputs(batch)   # already [0,1]-scale tensors
            x_dev, y_dev = x.to(device), y.to(device)
            pred = model(x_dev).cpu().numpy()
            y_np = y_dev.cpu().numpy()
            cloudy_np = batch["s2_cloudy"][:, LISS4_IDX, :, :].numpy()  # already [0,1]

            for i in range(pred.shape[0]):
                p01 = clip01(pred[i])
                t01 = clip01(y_np[i])
                c01 = clip01(cloudy_np[i])

                all_psnr.append(compute_psnr_safe(p01, t01))
                all_ssim.append(sk_ssim(to_uint8_hwc(t01), to_uint8_hwc(p01), data_range=255, channel_axis=2))
                all_sam.append(compute_sam(pred[i], y_np[i]))
                input_mse.append(np.mean((c01 - t01) ** 2))

                if saved_samples < args.num_samples:
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    axes[0].imshow(to_uint8_hwc(c01)); axes[0].set_title("Cloudy Input"); axes[0].axis("off")
                    axes[1].imshow(to_uint8_hwc(p01)); axes[1].set_title("Reconstructed"); axes[1].axis("off")
                    axes[2].imshow(to_uint8_hwc(t01)); axes[2].set_title("Clean (GT)"); axes[2].axis("off")
                    plt.tight_layout()
                    fig.savefig(out_dir / f"sample_{saved_samples+1}.png", dpi=150)
                    plt.close(fig)
                    saved_samples += 1

    all_psnr, all_ssim, all_sam, input_mse = map(np.array, (all_psnr, all_ssim, all_sam, input_mse))

    print("\n=== Input MSE distribution (cloudy vs clean, [0,1] scale) ===")
    for pct in [10, 25, 50, 75, 90, 99]:
        print(f"  p{pct}: {np.percentile(input_mse, pct):.6f}")

    print(f"\n=== OVERALL ({len(all_psnr)} patches) ===")
    print(f"PSNR: {all_psnr.mean():.3f} dB   SSIM: {all_ssim.mean():.4f}   SAM: {all_sam.mean():.3f} deg")
    print(f"\nSample comparison images saved to: {out_dir}")


if __name__ == "__main__":
    main()