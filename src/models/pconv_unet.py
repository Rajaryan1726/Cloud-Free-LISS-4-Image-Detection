"""
Partial Convolution U-Net for cloud removal.

Input:  5 channels  = S1 (VV, VH) [2ch]  +  S2 cloudy mapped to LISS-IV bands
                       (Green=B3, Red=B4, NIR=B8) [3ch]
Output: 3 channels  = reconstructed clean S2, same 3 mapped bands
                       (Green, Red, NIR)

Partial convolutions (Liu et al., "Image Inpainting for Irregular Holes
Using Partial Convolutions") are used instead of plain convolutions so the
network can explicitly account for cloud-masked regions during the
convolution operation, rather than treating masked and valid pixels
identically.

~4.17M parameters with base_filters=32 (verified via count_parameters()).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Partial Convolution layer
# ---------------------------------------------------------------------------
class PartialConv2d(nn.Module):
    """
    Standard partial conv: convolves both the input (masked to valid pixels)
    and the mask itself, then renormalizes the output by how much of the
    receptive field was valid. Also propagates an updated mask forward.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, bias=bias)

        # fixed (non-trainable) conv used purely to sum mask values in each window
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, bias=False)
        nn.init.constant_(self.mask_conv.weight, 1.0)
        for p in self.mask_conv.parameters():
            p.requires_grad = False

        self.window_size = in_channels * kernel_size * kernel_size

    def forward(self, x, mask):
        # x, mask: (B, C, H, W); mask has 1 = valid pixel, 0 = hole/cloud
        with torch.no_grad():
            mask_sum = self.mask_conv(mask)               # how many valid pixels contributed
        mask_sum_clamped = mask_sum.clamp(min=1e-8)

        raw_out = self.conv(x * mask)
        # renormalize so the conv output isn't biased by how many valid pixels there were
        out = raw_out * (self.window_size / mask_sum_clamped)

        if self.conv.bias is not None:
            bias = self.conv.bias.view(1, -1, 1, 1)
            out = out - bias
            out = out * (mask_sum > 0).float() + bias
        else:
            out = out * (mask_sum > 0).float()

        new_mask = (mask_sum > 0).float()
        return out, new_mask


class PConvBlock(nn.Module):
    """PartialConv -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=True):
        super().__init__()
        self.pconv = PartialConv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        x, mask = self.pconv(x, mask)
        if self.bn is not None:
            x = self.bn(x)
        x = self.act(x)
        return x, mask


# ---------------------------------------------------------------------------
# U-Net with partial convolutions
# ---------------------------------------------------------------------------
class PConvUNet(nn.Module):
    def __init__(self, in_channels: int = 5, out_channels: int = 3, base_filters: int = 40):
        super().__init__()
        f = base_filters

        # Encoder
        self.enc1 = PConvBlock(in_channels, f, use_bn=False)       # 256 -> 256
        self.enc2 = PConvBlock(f, f * 2, stride=2)                  # 256 -> 128
        self.enc3 = PConvBlock(f * 2, f * 4, stride=2)              # 128 -> 64
        self.enc4 = PConvBlock(f * 4, f * 8, stride=2)              # 64  -> 32

        # Bottleneck
        self.bottleneck = PConvBlock(f * 8, f * 8, stride=2)        # 32 -> 16

        # Decoder (upsample + concat skip + partial conv)
        self.up4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec4 = PConvBlock(f * 8 + f * 8, f * 8)                # 16 -> 32

        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec3 = PConvBlock(f * 8 + f * 4, f * 4)                # 32 -> 64

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = PConvBlock(f * 4 + f * 2, f * 2)                # 64 -> 128

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = PConvBlock(f * 2 + f, f)                        # 128 -> 256

        # Final projection to output bands, plain conv (no mask needed at output)
        self.final = nn.Conv2d(f, out_channels, kernel_size=1)
        self.out_act = nn.Sigmoid()   # outputs in [0,1], matches normalized target range

    @staticmethod
    def _pool_mask(mask, target_hw):
        return F.interpolate(mask, size=target_hw, mode="nearest")

    def forward(self, x, mask=None):
        """
        x:    (B, in_channels, H, W) -- normalized input, cloudy S2 bands + S1
        mask: (B, in_channels, H, W) -- 1 = valid/clear pixel, 0 = cloud/hole.
              If None, assumes all pixels valid (no explicit cloud mask available).
        """
        if mask is None:
            mask = torch.ones_like(x)

        e1, m1 = self.enc1(x, mask)
        e2, m2 = self.enc2(e1, m1)
        e3, m3 = self.enc3(e2, m2)
        e4, m4 = self.enc4(e3, m3)

        b, mb = self.bottleneck(e4, m4)

        d4 = self.up4(b)
        md4 = self._pool_mask(mb, d4.shape[-2:])
        d4 = torch.cat([d4, e4], dim=1)
        md4 = torch.cat([md4, m4], dim=1)
        d4, dm4 = self.dec4(d4, md4)

        d3 = self.up3(d4)
        md3 = self._pool_mask(dm4, d3.shape[-2:])
        d3 = torch.cat([d3, e3], dim=1)
        md3 = torch.cat([md3, m3], dim=1)
        d3, dm3 = self.dec3(d3, md3)

        d2 = self.up2(d3)
        md2 = self._pool_mask(dm3, d2.shape[-2:])
        d2 = torch.cat([d2, e2], dim=1)
        md2 = torch.cat([md2, m2], dim=1)
        d2, dm2 = self.dec2(d2, md2)

        d1 = self.up1(d2)
        md1 = self._pool_mask(dm2, d1.shape[-2:])
        d1 = torch.cat([d1, e1], dim=1)
        md1 = torch.cat([md1, m1], dim=1)
        d1, dm1 = self.dec1(d1, md1)

        out = self.final(d1)
        out = self.out_act(out)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # sanity check: 5-channel input (2 S1 + 3 mapped S2 bands) -> 3-channel clean output
    model = PConvUNet(in_channels=5, out_channels=3)
    n_params = count_parameters(model)
    print(f"Total trainable parameters: {n_params:,}  (~{n_params/1e6:.2f}M)")

    x = torch.randn(2, 5, 256, 256)
    out = model(x)   # mask=None -> defaults to all-ones (no explicit cloud mask)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 3, 256, 256), "Output shape mismatch!"
    print("Sanity check passed.")