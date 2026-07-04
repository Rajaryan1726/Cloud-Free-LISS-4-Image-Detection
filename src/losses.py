"""
Combined loss for cloud-removal training: L1 + SSIM + SAM.

- L1: per-pixel reconstruction accuracy (robust to outliers vs L2)
- SSIM: structural/spatial similarity (preserves edges, textures)
- SAM: Spectral Angle Mapper -- measures the angle between predicted and
       target spectral vectors at each pixel, independent of brightness.
       Important for satellite imagery where spectral consistency across
       bands matters as much as spatial accuracy.

Usage:
    criterion = CombinedLoss(l1_weight=1.0, ssim_weight=1.0, sam_weight=0.1)
    loss, parts = criterion(pred, target)   # pred, target: (B, C, H, W), values in [0,1]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SSIM (windowed, computed per-channel then averaged -- works for any C, not
# just 3-channel RGB, so it's fine for the 13-band S2 stack too)
# ---------------------------------------------------------------------------
def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)              # (1, window_size)
    window_2d = g.T @ g                          # (window_size, window_size)
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self._window_cache = {}   # keyed by (channels, device, dtype) to avoid rebuilding every call

    def _get_window(self, channels, device, dtype):
        key = (channels, device, dtype)
        if key not in self._window_cache:
            self._window_cache[key] = _gaussian_window(
                self.window_size, self.sigma, channels, device, dtype
            )
        return self._window_cache[key]

    def forward(self, pred, target):
        # pred, target expected in [0,1]
        C = pred.shape[1]
        window = self._get_window(C, pred.device, pred.dtype)
        pad = self.window_size // 2

        mu_pred = F.conv2d(pred, window, padding=pad, groups=C)
        mu_target = F.conv2d(target, window, padding=pad, groups=C)

        mu_pred_sq = mu_pred.pow(2)
        mu_target_sq = mu_target.pow(2)
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu_pred_sq
        sigma_target_sq = F.conv2d(target * target, window, padding=pad, groups=C) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, window, padding=pad, groups=C) - mu_pred_target

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / (
            (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
        )
        ssim_val = ssim_map.mean()
        return 1.0 - ssim_val   # loss: lower is better


# ---------------------------------------------------------------------------
# SAM (Spectral Angle Mapper)
# ---------------------------------------------------------------------------
class SAMLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # pred, target: (B, C, H, W) -> treat channel dim as the spectral vector per pixel
        dot = (pred * target).sum(dim=1)                          # (B, H, W)
        pred_norm = pred.norm(dim=1).clamp(min=self.eps)
        target_norm = target.norm(dim=1).clamp(min=self.eps)
        cos_angle = (dot / (pred_norm * target_norm)).clamp(-1 + 1e-7, 1 - 1e-7)
        angle = torch.acos(cos_angle)                              # radians, per pixel
        return angle.mean()


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------
class CombinedLoss(nn.Module):
    def __init__(self, l1_weight: float = 1.0, ssim_weight: float = 1.0, sam_weight: float = 0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.sam_weight = sam_weight

        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.sam = SAMLoss()

    def forward(self, pred, target):
        l1_val = self.l1(pred, target)
        ssim_val = self.ssim(pred, target)
        sam_val = self.sam(pred, target)

        total = (
            self.l1_weight * l1_val
            + self.ssim_weight * ssim_val
            + self.sam_weight * sam_val
        )

        parts = {
            "l1": l1_val.item(),
            "ssim": ssim_val.item(),
            "sam": sam_val.item(),
            "total": total.item(),
        }
        return total, parts


if __name__ == "__main__":
    # sanity check
    torch.manual_seed(0)
    pred = torch.rand(2, 3, 64, 64)
    target = torch.rand(2, 3, 64, 64)

    criterion = CombinedLoss()
    loss, parts = criterion(pred, target)
    print("Combined loss:", loss.item())
    print("Breakdown:", parts)

    # identical inputs -> loss should be near zero
    loss_same, parts_same = criterion(pred, pred)
    print("\nSanity check (pred vs pred, should be ~0):")
    print("Combined loss:", loss_same.item())
    print("Breakdown:", parts_same)