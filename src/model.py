"""Compact Restormer-style Transformer for single-image motion deblurring.

Design follows the building blocks of Restormer (Zamir et al., CVPR 2022):
  * MDTA  - multi-Dconv head TRANSPOSED attention: attention is computed between CHANNELS, not between
            pixels, so its cost grows linearly (not quadratically) with the image size.
  * GDFN  - gated-Dconv feed-forward network: two parallel branches multiplied element-wise (one through GELU).
  * a U-shaped encoder-decoder with skip connections, and a RESIDUAL output: restored = input + prediction.
This is our own small re-implementation (few channels, few blocks), written for this course project; it is
NOT a new architecture. Compactness is controlled by CONFIGS below and measured with count_parameters().
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Three sizes for the parameter-budget ablation. dim = channels at the first level (doubles at each level).
# blocks = number of Transformer blocks at (level 1, level 2, level 3 / latent, final refinement).
CONFIGS = {
    "tiny": dict(dim=16, blocks=(1, 1, 2, 1)),
    "small": dict(dim=24, blocks=(2, 2, 3, 1)),
    "base": dict(dim=32, blocks=(2, 3, 4, 2)),
}
HEADS = (1, 2, 4)  # attention heads at levels 1, 2, 3 (channels per level: dim, 2*dim, 4*dim)


class LayerNorm2d(nn.Module):
    """Normalises over the channel dimension at every pixel of a (B, C, H, W) tensor."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class MDTA(nn.Module):
    """Multi-Dconv head transposed attention (attention between channels)."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by the number of heads"
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))  # learnable scale of the attention logits
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)  # 1x1 conv: mixes channels
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=False)  # depth-wise: adds local context
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)
        # (B, C, H, W) -> (B, heads, C/heads, H*W): every channel becomes one "token" of length H*W
        q = q.reshape(b, self.heads, c // self.heads, h * w)
        k = k.reshape(b, self.heads, c // self.heads, h * w)
        v = v.reshape(b, self.heads, c // self.heads, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # (B, heads, C/heads, C/heads): channel-to-channel
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(b, c, h, w)
        return self.proj(out)


class GDFN(nn.Module):
    """Gated-Dconv feed-forward network."""

    def __init__(self, dim: int, expansion: float = 2.66):
        super().__init__()
        hidden = int(dim * expansion)
        self.proj_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=False)
        self.dw = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1, groups=hidden * 2, bias=False)
        self.proj_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.dw(self.proj_in(x)).chunk(2, dim=1)
        return self.proj_out(F.gelu(x1) * x2)  # gating: one branch controls what the other lets through


class TransformerBlock(nn.Module):
    """norm -> MDTA -> add (skip) ; norm -> GDFN -> add (skip)."""

    def __init__(self, dim: int, heads: int, expansion: float = 2.66):
        super().__init__()
        self.norm1, self.attn = LayerNorm2d(dim), MDTA(dim, heads)
        self.norm2, self.ffn = LayerNorm2d(dim), GDFN(dim, expansion)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


def blocks(dim: int, heads: int, n: int) -> nn.Sequential:
    return nn.Sequential(*[TransformerBlock(dim, heads) for _ in range(n)])


class Downsample(nn.Module):
    """Halves H and W, doubles channels (conv to dim/2, then PixelUnshuffle x2 -> 2*dim)."""

    def __init__(self, dim: int):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(dim, dim // 2, 3, padding=1, bias=False), nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """Doubles H and W, halves channels (conv to 2*dim, then PixelShuffle x2 -> dim/2)."""

    def __init__(self, dim: int):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(dim, dim * 2, 3, padding=1, bias=False), nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class CompactRestormer(nn.Module):
    def __init__(self, dim: int = 24, blocks_per_level=(2, 2, 3, 1)):
        super().__init__()
        n1, n2, n3, n_ref = blocks_per_level
        h1, h2, h3 = HEADS
        self.embed = nn.Conv2d(3, dim, 3, padding=1, bias=False)
        self.enc1 = blocks(dim, h1, n1)
        self.down1 = Downsample(dim)
        self.enc2 = blocks(dim * 2, h2, n2)
        self.down2 = Downsample(dim * 2)
        self.latent = blocks(dim * 4, h3, n3)
        self.up2 = Upsample(dim * 4)
        self.fuse2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=False)  # after concatenating the skip connection
        self.dec2 = blocks(dim * 2, h2, n2)
        self.up1 = Upsample(dim * 2)
        self.fuse1 = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.dec1 = blocks(dim, h1, n1)
        self.refine = blocks(dim, h1, n_ref)
        self.out = nn.Conv2d(dim, 3, 3, padding=1, bias=False)

    def forward(self, x):
        h, w = x.shape[-2:]
        ph, pw = (-h) % 4, (-w) % 4  # two downsamplings need H and W divisible by 4
        inp = F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x
        e1 = self.enc1(self.embed(inp))
        e2 = self.enc2(self.down1(e1))
        z = self.latent(self.down2(e2))
        d2 = self.dec2(self.fuse2(torch.cat([self.up2(z), e2], dim=1)))
        d1 = self.dec1(self.fuse1(torch.cat([self.up1(d2), e1], dim=1)))
        out = self.out(self.refine(d1))
        return (inp + out)[..., :h, :w]  # residual output: input + learned correction


def build_model(name: str) -> CompactRestormer:
    cfg = CONFIGS[name]
    return CompactRestormer(dim=cfg["dim"], blocks_per_level=cfg["blocks"])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
