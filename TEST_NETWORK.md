# Anonymized vela-friendly grouped-conv packed ensemble (N=34) — deployed eager architecture

```python
#!/usr/bin/env python3
"""Anonymized vela-friendly (grouped-conv packed) ensemble — eager architecture.

This reflects the DEPLOYED / Ethos-U-friendly form, i.e. the architecture AFTER
the two export-time transforms ("surgery"):

  1. Permute removal — channel-first normalization (no transpose/MoveAxis ops),
     and the channel-last Linears inside each conv block become 1x1 convs.
     (Transposes break NPU subgraph fusion; 1x1 conv == channel-last Linear.)

  2. Ensemble packing — the N independently-trained members are fused into GROUPED
     convolutions (groups = N), so all members run in a single command stream.
     The input window is replicated across the channel dim to feed the grouped stem.

Two packing modes (both parametric — nothing is unrolled by hand):
  * full pack:    one grouped tree with groups = N   (peak int32 width = N*C).
  * semi-grouped: members split into balanced chunks (<= group_size); each chunk
                  is one grouped tree, run sequentially. Bounds peak SRAM / int32
                  width to group_size*C. N=34, group_size=8 -> chunks [7,7,7,7,6].

The conv "tree" is packed; each member's pooling + FC head stays separate (every
member is independently trained and has its own head). Combination is a mean of
per-member deltas plus a linear reference calibration.

Architecture-faithful, RANDOMLY INITIALIZED (no trained weights), generic names.
In the real export the grouped-conv weights are the concatenation of the N
members' trained weights; here they're random since this is an architecture repro.

Runnable: `python3 anon_packed_ensemble.py` (needs a torch environment).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class MemberConfig:
    in_h: int = 20            # input window height
    in_w: int = 48            # input window width
    channels: int = 32        # encoder width (constant through the tree)
    num_blocks: int = 3       # residual blocks after the stem
    stem_kernel: tuple[int, int] = (20, 1)   # collapses H (20 -> 1), keeps W
    block_kernel: tuple[int, int] = (1, 7)   # depthwise spatial kernel
    block_expand: int = 2     # block MLP expansion (C -> 2C -> C)
    layer_scale_init: float = 3e-6
    n_out: int = 2            # regression targets
    mlp_hidden: int = 256
    add_ref: bool = True      # concat the [B, n_out] reference scalar into the head
    manual_features: int = 2  # flattened aux features concatenated into the head


# ---------------------------------------------------------------------------
# Channel-first building blocks (post permute-removal surgery)
# ---------------------------------------------------------------------------
class ChannelNorm(nn.Module):
    """Channel-first LayerNorm, grouped for n_groups members (no permute ops).

    Normalizes each member's `channels`-wide block independently inside one
    (B, n_groups*channels, H, W) tensor. n_groups=1 is the single-member form;
    n_groups=N is the packed form.
    """

    def __init__(self, n_groups: int, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.n_groups = n_groups
        self.channels = channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_groups * channels))
        self.bias = nn.Parameter(torch.zeros(n_groups * channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, nc, h, w = x.shape
        xg = x.view(b, self.n_groups, self.channels, h, w)
        mean = xg.mean(dim=2, keepdim=True)
        var = ((xg - mean) ** 2).mean(dim=2, keepdim=True)
        xg = (xg - mean) * torch.rsqrt(var + self.eps)
        x = xg.reshape(b, nc, h, w)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class DepthwiseBlock(nn.Module):
    """ConvNeXt-style residual block, channel-first, grouped for n_groups members.

    depthwise conv (groups=n_groups*C) -> ChannelNorm -> 1x1 conv (expand, groups
    =n_groups) -> GELU -> 1x1 conv (project, groups=n_groups) -> layer-scale ->
    residual. The two 1x1 convs are the channel-last Linears after permute removal.
    """

    def __init__(self, n_groups: int, cfg: MemberConfig) -> None:
        super().__init__()
        c = cfg.channels
        hidden = c * cfg.block_expand
        self.dw = nn.Conv2d(
            n_groups * c, n_groups * c, cfg.block_kernel,
            padding="same", groups=n_groups * c,
        )
        self.norm = ChannelNorm(n_groups, c)
        self.pw1 = nn.Conv2d(n_groups * c, n_groups * hidden, 1, groups=n_groups)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(n_groups * hidden, n_groups * c, 1, groups=n_groups)
        self.layer_scale = nn.Parameter(
            torch.ones(n_groups * c, 1, 1) * cfg.layer_scale_init
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dw(x)
        out = self.norm(out)
        out = self.pw2(self.act(self.pw1(out)))
        out = self.layer_scale * out
        return out + x


class PackedConvTree(nn.Module):
    """Grouped-conv encoder for n_groups members fused together.

    Input  (B, n_groups, H, W)  -> Output (B, n_groups*C, 1, W).
    n_groups=1 is a single member's encoder; n_groups=K fuses K members.
    """

    def __init__(self, n_groups: int, cfg: MemberConfig) -> None:
        super().__init__()
        c = cfg.channels
        self.stem = nn.Conv2d(n_groups * 1, n_groups * c, cfg.stem_kernel, groups=n_groups)
        self.stem_norm = ChannelNorm(n_groups, c)
        self.blocks = nn.ModuleList(
            [DepthwiseBlock(n_groups, cfg) for _ in range(cfg.num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_norm(self.stem(x))
        for blk in self.blocks:
            x = blk(x)
        return x


class WeightedMeanPool(nn.Module):
    """Learned weighted mean over the last dim, driven by channel 0: (B,C,L)->(B,C)."""

    def __init__(self, length: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(x[:, 0:1, :] + self.bias, dim=-1)
        return (x * w).sum(dim=-1)


class MemberHead(nn.Module):
    """Per-member pooling + FC head (NOT packed; each member has its own)."""

    def __init__(self, cfg: MemberConfig) -> None:
        super().__init__()
        c = cfg.channels
        self.tgt_pool = WeightedMeanPool(cfg.in_w)
        self.ref_pool = WeightedMeanPool(cfg.in_w)
        mlp_in = 2 * c + (cfg.n_out if cfg.add_ref else 0) + cfg.manual_features
        self.penultimate = nn.Linear(mlp_in, cfg.mlp_hidden)
        self.output = nn.Linear(cfg.mlp_hidden, cfg.n_out)
        self.add_ref = cfg.add_ref

    def forward(self, tgt_feat, ref_feat, ref_scalar, aux_feat) -> torch.Tensor:
        # tgt_feat/ref_feat: (B, C, 1, W) -> mean over H -> (B, C, W) -> pool -> (B, C)
        t = self.tgt_pool(tgt_feat.mean(dim=2))
        r = self.ref_pool(ref_feat.mean(dim=2))
        parts = [t, r]
        if self.add_ref:
            parts.append(ref_scalar)
        parts.append(aux_feat)
        embed = torch.cat(parts, dim=1)
        pen = torch.relu(self.penultimate(embed))
        return self.output(pen)  # per-member delta (B, n_out)


def balanced_group_sizes(n: int, max_group: int) -> list[int]:
    """Split N members into balanced chunks no larger than max_group.

    N=34, max_group=8 -> [7, 7, 7, 7, 6] (5 chunks), not greedy [8,8,8,8,2].
    """
    num = (n + max_group - 1) // max_group
    base, rem = divmod(n, num)
    return [base + 1] * rem + [base] * (num - rem)


class PackedEnsemble(nn.Module):
    """Vela-friendly packed ensemble of N members.

    group_size is None or >= N  -> one grouped tree (full pack, groups=N).
    group_size <  N             -> balanced chunks, each a grouped tree run
                                   sequentially (semi-grouped; peak width = group_size*C).
    """

    def __init__(
        self, num_members: int, cfg: MemberConfig, group_size: int | None = None
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_members = num_members
        gs = group_size if group_size is not None else num_members
        self.group_sizes = balanced_group_sizes(num_members, gs)
        self.trees = nn.ModuleList([PackedConvTree(k, cfg) for k in self.group_sizes])
        self.heads = nn.ModuleList([MemberHead(cfg) for _ in range(num_members)])
        # Frozen linear calibration of the reference scalar (identity by default).
        self.register_buffer("A", torch.eye(cfg.n_out))
        self.register_buffer("b", torch.zeros(cfg.n_out))

    def forward(self, window_a, window_b, ref_scalar, aux_feat) -> torch.Tensor:
        cfg = self.cfg
        aline_ref = ref_scalar @ self.A.T + self.b
        deltas = []
        gi = 0  # global member index (selects that member's head)
        for tree, k in zip(self.trees, self.group_sizes):
            tgt = window_a.unsqueeze(1).repeat(1, k, 1, 1)  # (B, k, H, W)
            ref = window_b.unsqueeze(1).repeat(1, k, 1, 1)
            tgt = tree(tgt)  # (B, k*C, 1, W)
            ref = tree(ref)
            b_, _, h_, w_ = tgt.shape
            tgt_u = tgt.view(b_, k, cfg.channels, h_, w_)
            ref_u = ref.view(b_, k, cfg.channels, h_, w_)
            for i in range(k):
                deltas.append(
                    self.heads[gi](tgt_u[:, i], ref_u[:, i], aline_ref, aux_feat)
                )
                gi += 1
        mean_delta = torch.stack(deltas, dim=0).mean(dim=0)
        return mean_delta + aline_ref  # absolute prediction (B, n_out)


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = MemberConfig()
    N = 34
    GROUP_SIZE = 8  # semi-grouped chunks of <=8 -> [7,7,7,7,6]; set None for full pack

    model = PackedEnsemble(N, cfg, group_size=GROUP_SIZE).eval()
    print(model)
    print(f"\nmembers: {N}   group sizes: {model.group_sizes}")
    print(f"total params: {sum(p.numel() for p in model.parameters()):,}")

    B = 1
    wa = torch.randn(B, cfg.in_h, cfg.in_w)
    wb = torch.randn(B, cfg.in_h, cfg.in_w)
    ref = torch.randn(B, cfg.n_out)
    feat = torch.randn(B, cfg.manual_features)
    with torch.no_grad():
        y = model(wa, wb, ref, feat)
    print(f"\ninputs: windows {tuple(wa.shape)} x2, ref {tuple(ref.shape)}, aux {tuple(feat.shape)}")
    print(f"output: {tuple(y.shape)} -> {y.tolist()}")
```
