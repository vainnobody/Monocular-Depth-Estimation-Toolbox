# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
"""
Minimal DinoViT implementation extracted from this repo to ease migration.

It keeps only the Vision Transformer backbone and the official pretrained
weight loader so you can drop the file into another project without pulling
the whole training stack.
"""

from __future__ import annotations

import logging
import math
import os.path as osp
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse
from mmengine.logging import print_log
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
# from ..builder import BACKBONES
DINOV3_BASE_URL = "https://dl.fbaipublicfiles.com/dinov3"
logger = logging.getLogger("dinov3_minimal")
import torch.utils.checkpoint as cp
import math
from torch import Tensor
try:
    from mmrotate.registry import MODELS
except ImportError:  # pragma: no cover - depth-only environments may not install mmrotate.
    class _DummyRegistry:
        def register_module(self, *args, **kwargs):
            def decorator(obj):
                return obj
            return decorator

    MODELS = _DummyRegistry()

# =========================================================
# 1) LoRA-style expert
# =========================================================
class LoRAExpert(nn.Module):
    """
    Pure low-rank linear expert:
        x -> proj_down -> proj_up

    Most parameter-efficient.
    Closest to standard LoRA-style delta branch.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        r: int = 32,
        use_scale: bool = True,
        device=None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.r = r

        self.proj_down = nn.Linear(in_dim, r, bias=False, device=device)
        self.proj_up = nn.Linear(r, out_dim, bias=False, device=device)

        if use_scale:
            self.scale = nn.Parameter(torch.ones(1, device=device))
        else:
            self.register_buffer("scale", torch.ones(1, device=device), persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.proj_down.weight, nonlinearity="linear")
        nn.init.zeros_(self.proj_up.weight)

    def zero_init_output(self):
        nn.init.zeros_(self.proj_up.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj_up(self.proj_down(x)) * self.scale


# =========================================================
# 2) Adapter-style expert
# =========================================================
class AdapterExpert(nn.Module):
    """
    Bottleneck adapter-like expert:
        x -> proj_down -> GELU -> Dropout -> proj_up

    Slightly larger than LoRA-style, but usually stronger.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        r: int = 64,
        dropout: float = 0.1,
        act_layer=nn.GELU,
        device=None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.r = r

        self.proj_down = nn.Linear(in_dim, r, bias=False, device=device)
        self.act = act_layer()
        self.dropout = nn.Dropout(dropout)
        self.proj_up = nn.Linear(r, out_dim, bias=False, device=device)


    def forward(self, x: Tensor) -> Tensor:
        x = self.proj_down(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.proj_up(x)
        return x


# =========================================================
# 3) Conv-style expert
# Note:
#   This version supports [B, N, C] input directly.
#   If input is [T, C], it will be treated as one pseudo-batch.
#   Best used when x is token sequence from ViT.
# =========================================================
class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, device=None):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=True,
            device=device,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            device=device,
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.depthwise.weight, a=math.sqrt(5))
        nn.init.kaiming_normal_(self.pointwise.weight, a=math.sqrt(5))
        nn.init.constant_(self.depthwise.bias, 0)
        nn.init.constant_(self.pointwise.bias, 0)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ConvTokenMixer(nn.Module):
    """
    Lightweight token-grid conv expert in bottleneck space.

    Assumes token layout is [B, N, C] and N is (num_prefix_tokens + H*W).
    """
    def __init__(
        self,
        r: int,
        *,
        num_prefix_tokens: int = 0,
        kernel_size: int = 3,
        use_norm: bool = True,
        activate: str = "gelu",
        scale: float = 2.0,
        dropout: float = 0.1,
        device=None,
    ):
        super().__init__()
        self.r = r
        self.num_prefix_tokens = num_prefix_tokens
        self.scale = scale
        self.use_norm = use_norm
        self.dropout = nn.Dropout(dropout)

        if activate == "gelu":
            self.act = nn.GELU()
        elif activate == "relu":
            self.act = nn.ReLU()
        elif activate == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activate: {activate}")

        self.conv1 = DWConv(
            r, r, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, device=device
        )
        self.conv2 = DWConv(
            r, r, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, device=device
        )

        if use_norm:
            self.norm1 = nn.LayerNorm(r, device=device)
            self.norm2 = nn.LayerNorm(r, device=device)

        self._init_parameters()

    def _init_parameters(self):
        if self.use_norm:
            nn.init.ones_(self.norm1.weight)
            nn.init.zeros_(self.norm1.bias)
            nn.init.ones_(self.norm2.weight)
            nn.init.zeros_(self.norm2.bias)

    def forward(self, x: Tensor, scale: Optional[float] = None) -> Tensor:
        # x: [B, N, r]
        if x.dim() != 3:
            raise ValueError(f"ConvTokenMixer expects [B, N, C], got {tuple(x.shape)}")

        if scale is None:
            scale = self.scale

        if self.num_prefix_tokens > 0:
            prefix = x[:, :self.num_prefix_tokens, :]
            x = x[:, self.num_prefix_tokens:, :]
        else:
            prefix = None

        B, N, r = x.shape
        H = W = int(N ** 0.5)
        if H * W != N:
            raise ValueError(
                f"ConvTokenMixer expects square token grid after prefix removal, got N={N}"
            )

        x_2d = x.permute(0, 2, 1).reshape(B, r, H, W).contiguous()

        x_scale1 = F.interpolate(x_2d, scale_factor=scale, mode="bilinear", align_corners=False)
        x_conv1 = self.dropout(self.act(self.conv1(x_scale1)))
        x_conv1 = F.interpolate(x_conv1, size=(H, W), mode="bilinear", align_corners=False)

        x_scale2 = F.interpolate(x_2d, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        x_conv2 = self.dropout(self.act(self.conv2(x_scale2)))
        x_conv2 = F.interpolate(x_conv2, size=(H, W), mode="bilinear", align_corners=False)

        if self.use_norm:
            x_conv1 = x_conv1.permute(0, 2, 3, 1).contiguous()
            x_conv2 = x_conv2.permute(0, 2, 3, 1).contiguous()
            x_conv1 = self.norm1(x_conv1)
            x_conv2 = self.norm2(x_conv2)
            x_conv1 = x_conv1.permute(0, 3, 1, 2).contiguous()
            x_conv2 = x_conv2.permute(0, 3, 1, 2).contiguous()

        x = (x_conv1 + x_conv2) / 2.0
        x = x.reshape(B, r, N).permute(0, 2, 1).contiguous()

        if prefix is not None:
            x = torch.cat([prefix, x], dim=1)

        return x


class ConvExpert(nn.Module):
    """
    Conv-style lightweight expert:
        x -> proj_down -> token-mixer conv -> proj_up

    Best for [B, N, C].
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        r: int = 64,
        dropout: float = 0.1,
        num_prefix_tokens: int = 0,
        kernel_size: int = 3,
        use_norm: bool = True,
        activate: str = "gelu",
        conv_scale: float = 2.0,
        use_scale: bool = True,
        device=None,
    ):
        super().__init__()
        self.proj_down = nn.Linear(in_dim, r, bias=False, device=device)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.conv = ConvTokenMixer(
            r=r,
            num_prefix_tokens=num_prefix_tokens,
            kernel_size=kernel_size,
            use_norm=use_norm,
            activate=activate,
            scale=conv_scale,
            dropout=dropout,
            device=device,
        )
        self.proj_up = nn.Linear(r, out_dim, bias=False, device=device)

        if use_scale:
            self.scale = nn.Parameter(torch.ones(1, device=device))
        else:
            self.register_buffer("scale", torch.ones(1, device=device), persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.proj_down.weight, nonlinearity="relu")
        nn.init.zeros_(self.proj_up.weight)

    def zero_init_output(self):
        nn.init.zeros_(self.proj_up.weight)

    def forward(self, x: Tensor) -> Tensor:
        original_ndim = x.dim()

        # Support [T, C] by reshaping to [1, T, C]
        if original_ndim == 2:
            x = x.unsqueeze(0)

        x = self.dropout(self.act(self.proj_down(x)))
        x = self.conv(x)
        x = self.proj_up(x) * self.scale

        if original_ndim == 2:
            x = x.squeeze(0)
        return x


# =========================================================
# Expert factory
# =========================================================
def build_expert(
    expert_type: str,
    in_dim: int,
    out_dim: int,
    *,
    expert_rank: int = 64,
    expert_dropout: float = 0.1,
    num_prefix_tokens: int = 0,
    conv_kernel_size: int = 3,
    conv_use_norm: bool = True,
    conv_activate: str = "gelu",
    conv_scale: float = 2.0,
    device=None,
):
    if expert_type == "lora":
        return LoRAExpert(
            in_dim=in_dim,
            out_dim=out_dim,
            r=expert_rank,
            use_scale=True,
            device=device,
        )
    elif expert_type == "adapter":
        return AdapterExpert(
            in_dim=in_dim,
            out_dim=out_dim,
            r=expert_rank,
            dropout=expert_dropout,
            device=device,
        )
    elif expert_type == "conv":
        return ConvExpert(
            in_dim=in_dim,
            out_dim=out_dim,
            r=expert_rank,
            dropout=expert_dropout,
            num_prefix_tokens=num_prefix_tokens,
            kernel_size=conv_kernel_size,
            use_norm=conv_use_norm,
            activate=conv_activate,
            conv_scale=conv_scale,
            use_scale=True,
            device=device,
        )
    else:
        raise ValueError(f"Unknown expert_type: {expert_type}")


# =========================================================
# Shared sparse top-k expert pool
# =========================================================
class _SharedPoolSparseExperts(nn.Module):
    """
    Shared sparse top-k experts with key-aware routing.
    key can be modality or task.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        keys=("opt", "sar"),
        num_experts: int = 4,
        top_k: int = 1,
        aux_loss_weight: float = 0.001,
        zero_init_experts: bool = True,
        zero_init_router: bool = False,
        expert_type: str = "adapter",   # "lora" | "adapter" | "conv"
        expert_rank: int = 64,
        expert_dropout: float = 0.1,
        num_prefix_tokens: int = 0,
        conv_kernel_size: int = 3,
        conv_use_norm: bool = True,
        conv_activate: str = "gelu",
        conv_scale: float = 2.0,
        device=None,
    ):
        super().__init__()
        assert num_experts >= 1
        assert 1 <= top_k <= num_experts

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.keys = tuple(keys)
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_weight = float(aux_loss_weight)
        self.expert_type = expert_type

        self.experts = nn.ModuleList([
            build_expert(
                expert_type=expert_type,
                in_dim=in_dim,
                out_dim=out_dim,
                expert_rank=expert_rank,
                expert_dropout=expert_dropout,
                num_prefix_tokens=num_prefix_tokens,
                conv_kernel_size=conv_kernel_size,
                conv_use_norm=conv_use_norm,
                conv_activate=conv_activate,
                conv_scale=conv_scale,
                device=device,
            )
            for _ in range(num_experts)
        ])

        self.routers = nn.ModuleDict({
            k: nn.Linear(in_dim, num_experts, bias=True, device=device)
            for k in self.keys
        })

        self.last_aux_loss = None

        if zero_init_experts:
            for expert in self.experts:
                if hasattr(expert, "zero_init_output"):
                    expert.zero_init_output()

        if zero_init_router:
            for router in self.routers.values():
                nn.init.zeros_(router.weight)
                if router.bias is not None:
                    nn.init.zeros_(router.bias)

    def _compute_aux_loss(self, probs_f: Tensor, topk_indices: Tensor, topk_scores: Tensor) -> Tensor:
        T, E = probs_f.shape
        wt_assign = probs_f.new_zeros(T, E)
        wt_assign.scatter_add_(1, topk_indices, topk_scores)
        f = wt_assign.mean(dim=0)
        p = probs_f.mean(dim=0)
        aux = E * torch.sum(f * p)
        return aux

    def forward(self, x: Tensor, *, key: str) -> Tensor:
        if key not in self.routers:
            raise KeyError(f"Unknown key='{key}'. Valid: {list(self.routers.keys())}")

        # router always works on flattened token dimension
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])  # [T, C]
        T = x_flat.shape[0]

        logits = self.routers[key](x_flat)
        probs = F.softmax(logits, dim=-1)

        topk_scores, topk_indices = torch.topk(probs, k=self.top_k, dim=-1)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        if self.training and self.aux_loss_weight > 0.0:
            self.last_aux_loss = self.aux_loss_weight * self._compute_aux_loss(
                probs, topk_indices, topk_scores
            )
        else:
            self.last_aux_loss = None

        out_flat = x_flat.new_zeros(T, self.out_dim)

        for e_idx, expert in enumerate(self.experts):
            rows, kpos = torch.where(topk_indices == e_idx)
            if rows.numel() > 0:
                x_sel = x_flat[rows]
                y_sel = expert(x_sel)
                gates = topk_scores[rows, kpos].unsqueeze(-1)
                out_flat[rows] += y_sel * gates

        # dummy path: keep all experts in graph
        dummy = x_flat.sum() * 0.0
        for expert in self.experts:
            for p in expert.parameters():
                dummy = dummy + p.sum() * 0.0

        out_flat = out_flat + dummy
        return out_flat.view(*original_shape, self.out_dim)


# =========================================================
# Final lightweight SharedPoolSparseMoEFC2
# =========================================================
class SharedPoolSparseMoEFC2(nn.Module):
    """
    fc2 replacement:
        y = shared(x) + alpha * modality_pool(x)

    只保留模态专家池，不再使用任务专家池。
    """
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        *,
        bias: bool = True,
        modalities=("opt", "sar"),
        num_experts: int = 4,
        top_k: int = 1,
        aux_loss_weight: float = 0.001,
        init_shared_from: Optional[nn.Linear] = None,
        zero_init_others: bool = True,
        zero_init_router: bool = False,
        alpha_init: float = 0.0001,
        learnable_alpha: bool = True,
        expert_type: str = "adapter",   # "lora" | "adapter" | "conv"
        expert_rank: int = 64,
        expert_dropout: float = 0.1,
        num_prefix_tokens: int = 0,
        device=None,
    ):
        super().__init__()
        self.modalities = tuple(modalities)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_weight = float(aux_loss_weight)
        self.expert_type = expert_type

        self.shared = nn.Linear(hidden_dim, out_dim, bias=bias, device=device)

        self.shared_pool_moe = _SharedPoolSparseExperts(
            in_dim=hidden_dim,
            out_dim=out_dim,
            keys=self.modalities,
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_weight=aux_loss_weight,
            zero_init_experts=zero_init_others,
            zero_init_router=zero_init_router,
            expert_type=expert_type,
            expert_rank=expert_rank,
            expert_dropout=expert_dropout,
            num_prefix_tokens=num_prefix_tokens,
            device=device,
        )

        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init), device=device))
        else:
            self.register_buffer(
                "alpha",
                torch.tensor(float(alpha_init), device=device),
                persistent=False
            )

        self.last_aux_loss = None

        if init_shared_from is not None:
            with torch.no_grad():
                self.shared.weight.copy_(init_shared_from.weight)
                if bias and init_shared_from.bias is not None and self.shared.bias is not None:
                    self.shared.bias.copy_(init_shared_from.bias)

    def get_aux_loss(self) -> Optional[Tensor]:
        return self.last_aux_loss

    def forward(self, x: Tensor, *, modality: Optional[str] = None, task: Optional[str] = None) -> Tensor:
        # task 参数保留，仅为兼容外部调用；内部不使用
        y = self.shared(x)

        if modality is not None:
            moe_y = self.shared_pool_moe(x, key=modality)
            if self.shared_pool_moe.last_aux_loss is not None:
                self.last_aux_loss = self.shared_pool_moe.last_aux_loss
            else:
                self.last_aux_loss = None
            y = y + self.alpha * moe_y
        else:
            self.last_aux_loss = None

        return y

# -----------------------------------------------------------------------------
# Small utility helpers
# -----------------------------------------------------------------------------
def cat_keep_shapes(x_list: List[Tensor]) -> Tuple[Tensor, List[Tuple[int, ...]], List[int]]:
    shapes = [x.shape for x in x_list]
    num_tokens = [x.select(dim=-1, index=0).numel() for x in x_list]
    flattened = torch.cat([x.flatten(0, -2) for x in x_list])
    return flattened, shapes, num_tokens


def uncat_with_shapes(flattened: Tensor, shapes: List[Tuple[int, ...]], num_tokens: List[int]) -> List[Tensor]:
    outputs_splitted = torch.split_with_sizes(flattened, num_tokens, dim=0)
    shapes_adjusted = [shape[:-1] + torch.Size([flattened.shape[-1]]) for shape in shapes]
    outputs_reshaped = [o.reshape(shape) for o, shape in zip(outputs_splitted, shapes_adjusted)]
    return outputs_reshaped


def named_apply(
    fn: Callable,
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_path = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_path,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


# -----------------------------------------------------------------------------
# Layers used by DinoViT
# -----------------------------------------------------------------------------
def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.empty(dim, device=device))
        self.init_values = init_values

    def reset_parameters(self):
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable | None = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

    def reset_parameters(self):
        k = 1 / (self.in_chans * (self.patch_size[0] ** 2))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def reset_parameters(self) -> None:
        nn.init.constant_(self.weight, 1)

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Norm2d(nn.Module):
    """
    Channel-wise LayerNorm applied on (B, C, H, W).
    """

    def __init__(self, embed_dim: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        x = self.ln(x)
        return x.permute(0, 3, 1, 2).contiguous()


class RopePositionEmbedding(nn.Module):
    """
    RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        self.dtype = dtype  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / max_HW  # [W]
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / min_HW  # [W]
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H  # [H]
            coords_w = torch.arange(0.5, W, **dd) / W  # [W]
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # Shift coords by adding a uniform value in [-shift, shift]
        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords += shift_hw[None, :]

        # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter_hw[None, :]

        # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords *= rescale_hw

        # Prepare angles and sin/cos
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)  # [HW, D]
        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2)
            )  # [D//4]
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, device=device, dtype=dtype)  # [D//4]
            periods = base**exponents  # range [1, max_period / min_period]
            periods = periods / base  # range [min_period / max_period, 1]
            periods = periods * self.max_period  # range [min_period, max_period]
        self.periods.data = periods


class ListForwardMixin(object):
    def forward(self, x: Tensor):
        raise NotImplementedError

    def forward_list(self, x_list: List[Tensor]) -> List[Tensor]:
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        x_flat = self.forward(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)



class Mlp(nn.Module, ListForwardMixin):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        device=None,
        # ---- MoE config ----
        use_moe_fc2: bool = True,
        moe_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.use_moe_fc2 = bool(use_moe_fc2)
        moe_cfg = moe_cfg or {}

        if self.use_moe_fc2:
            self.fc2 = SharedPoolSparseMoEFC2(
                hidden_dim=hidden_features,
                out_dim=out_features,
                bias=bias,
                modalities=moe_cfg.get("modalities", ("opt", "sar")),
                num_experts=moe_cfg.get("num_experts", 3),
                top_k=moe_cfg.get("top_k", 1),
                aux_loss_weight=moe_cfg.get("aux_loss_weight", 0.001),
                init_shared_from=moe_cfg.get("init_shared_from", None),
                zero_init_others=moe_cfg.get("zero_init_others", False),
                zero_init_router=moe_cfg.get("zero_init_router", False),
                alpha_init=moe_cfg.get("alpha_init", 0.0001),
                learnable_alpha=moe_cfg.get("learnable_alpha", True),
                expert_type=moe_cfg.get("expert_type", "adapter"),
                expert_rank=moe_cfg.get("expert_rank", 64),
                expert_dropout=moe_cfg.get("expert_dropout", drop),
                num_prefix_tokens=moe_cfg.get("num_prefix_tokens", 0),
                device=device,
            )
        else:
            self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, device=device)

        self.drop2 = nn.Dropout(drop)

    def forward(self, x: Tensor, *, modality: Optional[str] = None, task: Optional[str] = None) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        if isinstance(self.fc2, SharedPoolSparseMoEFC2):
            x = self.fc2(x, modality=modality, task=task)
        else:
            x = self.fc2(x)

        x = self.drop2(x)
        return x

    def forward_list(self, x_list: List[Tensor], *, modality_list=None, task_list=None) -> List[Tensor]:
        if modality_list is None:
            modality_list = [None] * len(x_list)
        if task_list is None:
            task_list = [None] * len(x_list)
        assert len(modality_list) == len(x_list)
        assert len(task_list) == len(x_list)

        outs = []
        for x, m, t in zip(x_list, modality_list, task_list):
            outs.append(self.forward(x, modality=m, task=t))
        return outs


class SwiGLUFFN(nn.Module, ListForwardMixin):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[Callable[..., nn.Module]] = None,
        drop: float = 0.0,
        bias: bool = True,
        align_to: int = 8,
        device=None,
        use_moe_fc2: bool = True,
        moe_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        moe_cfg = moe_cfg or {}

        d = int(hidden_features * 2 / 3)
        swiglu_hidden_features = d + (-d % align_to)
        self.w1 = nn.Linear(in_features, swiglu_hidden_features, bias=bias, device=device)
        self.w2 = nn.Linear(in_features, swiglu_hidden_features, bias=bias, device=device)
        self.drop = nn.Dropout(drop)
        self.use_moe_fc2 = bool(use_moe_fc2)

        if self.use_moe_fc2:
            self.w3 = SharedPoolSparseMoEFC2(
                hidden_dim=swiglu_hidden_features,
                out_dim=out_features,
                bias=bias,
                modalities=moe_cfg.get("modalities", ("opt", "sar")),
                num_experts=moe_cfg.get("num_experts", 4),
                top_k=moe_cfg.get("top_k", 1),
                aux_loss_weight=moe_cfg.get("aux_loss_weight", 0.001),
                init_shared_from=moe_cfg.get("init_shared_from", None),
                zero_init_others=moe_cfg.get("zero_init_others", False),
                zero_init_router=moe_cfg.get("zero_init_router", False),
                alpha_init=moe_cfg.get("alpha_init", 0.0001),
                learnable_alpha=moe_cfg.get("learnable_alpha", True),
                expert_type=moe_cfg.get("expert_type", "adapter"),
                expert_rank=moe_cfg.get("expert_rank", 64),
                expert_dropout=moe_cfg.get("expert_dropout", drop),
                num_prefix_tokens=moe_cfg.get("num_prefix_tokens", 0),
                device=device,
            )
        else:
            self.w3 = nn.Linear(swiglu_hidden_features, out_features, bias=bias, device=device)

    def forward(self, x: Tensor, *, modality: Optional[str] = None, task: Optional[str] = None) -> Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        hidden = self.drop(hidden)

        if isinstance(self.w3, SharedPoolSparseMoEFC2):
            return self.w3(hidden, modality=modality, task=task)
        return self.w3(hidden)

    def forward_list(self, x_list: List[Tensor], *, modality_list=None, task_list=None) -> List[Tensor]:
        if modality_list is None:
            modality_list = [None] * len(x_list)
        if task_list is None:
            task_list = [None] * len(x_list)
        assert len(modality_list) == len(x_list)
        assert len(task_list) == len(x_list)

        outs = []
        for x, m, t in zip(x_list, modality_list, task_list):
            outs.append(self.forward(x, modality=m, task=t))
        return outs


def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(input, self.weight, masked_bias)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rope(self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(self, x: Tensor, attn_bias=None, rope: Tensor = None) -> Tensor:
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x

    def forward_list(self, x_list, attn_bias=None, rope_list=None) -> List[Tensor]:
        assert len(x_list) == len(rope_list)
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for _, (qkv, _, rope) in enumerate(zip(qkv_list, shapes, rope_list)):
            att_out.append(self.compute_attention(qkv, attn_bias=attn_bias, rope=rope))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        sdp = getattr(torch.nn.functional, "scaled_dot_product_attention", None)
        if sdp is not None:
            x = sdp(q, k, v)
        else:
            # Torch < 2.0 compatibility: manual attention
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = torch.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        mask_k_bias: bool = False,
        device=None,
        with_cp=False,
        # with_cp=True,
        use_moe_fc2: bool = True,
        moe_cfg: Optional[Dict] = None
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
            use_moe_fc2=use_moe_fc2,
            moe_cfg=moe_cfg
        )
        self.ls2 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.sample_drop_ratio = drop_path
        self.with_cp = with_cp

    @staticmethod
    def _maybe_index_rope(rope: tuple[Tensor, Tensor] | None, indices: Tensor) -> tuple[Tensor, Tensor] | None:
        if rope is None:
            return None

        sin, cos = rope
        assert sin.ndim == cos.ndim
        if sin.ndim == 4:
            return sin[indices], cos[indices]  # [batch, heads, patches, embed_dim]
        else:
            return sin, cos  # [heads, patches, embed_dim] or [patches, embed_dim]

    def _forward_list(self, x_list: List[Tensor], rope_list=None, *, modality_list=None, task_list=None) -> List[Tensor]:
        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list]
        residual_scale_factors = [b / sample_subset_size for b, sample_subset_size in zip(b_list, sample_subset_sizes)]

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_1_list = [x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)]

            if rope_list is not None:
                rope_subset_list = [
                    self._maybe_index_rope(rope, indices_1) for rope, indices_1 in zip(rope_list, indices_1_list)
                ]
            else:
                rope_subset_list = rope_list

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
            residual_1_list = self.attn.forward_list(norm1, rope_list=rope_subset_list)

            x_attn_list = [
                torch.index_add(
                    x,
                    dim=0,
                    source=self.ls1(residual_1),
                    index=indices_1,
                    alpha=residual_scale_factor,
                )
                for x, residual_1, indices_1, residual_scale_factor in zip(
                    x_list, residual_1_list, indices_1_list, residual_scale_factors
                )
            ]

            indices_2_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_2_list = [x[indices_2] for x, indices_2 in zip(x_attn_list, indices_2_list)]
            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

            residual_2_list = self.mlp.forward_list(norm2_list, modality_list=modality_list, task_list=task_list)

            x_ffn = [
                torch.index_add(
                    x_attn,
                    dim=0,
                    source=self.ls2(residual_2),
                    index=indices_2,
                    alpha=residual_scale_factor,
                )
                for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                    x_attn_list, residual_2_list, indices_2_list, residual_scale_factors
                )
            ]
        else:
            x_out = []
            for i, (x, rope) in enumerate(zip(x_list, rope_list)):
                x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                n2 = self.norm2(x_attn)
                m = modality_list[i] if modality_list is not None else None
                t = task_list[i] if task_list is not None else None
                x_ffn = x_attn + self.ls2(self.mlp(n2, modality=m, task=t))
                x_out.append(x_ffn)
            x_ffn = x_out
        return x_ffn


    # def forward(self, x_or_x_list, rope_or_rope_list=None, *, modality=None, task=None, modality_list=None, task_list=None):
    #     if isinstance(x_or_x_list, Tensor):
    #         return self._forward_list(
    #             [x_or_x_list],
    #             rope_list=[rope_orf_rope_list],
    #             modality_list=[modality],
    #             task_list=[task],
    #         )[0]
    #     elif isinstance(x_or_x_list, list):
    #         if rope_or_rope_list is None:
    #             rope_or_rope_list = [None for _ in x_or_x_list]
    #         if modality_list is None and modality is not None:
    #             modality_list = [modality for _ in x_or_x_list]
    #         if task_list is None and task is not None:
    #             task_list = [task for _ in x_or_x_list]
    #         return self._forward_list(
    #             x_or_x_list,
    #             rope_list=rope_or_rope_list,
    #             modality_list=modality_list,
    #             task_list=task_list,
    #         )
    #     else:
    #         raise AssertionError


    def forward(self, x_or_x_list, rope_or_rope_list=None, *, modality=None, task=None, modality_list=None, task_list=None):
        if isinstance(x_or_x_list, Tensor):
            x = x_or_x_list
            rope = rope_or_rope_list

            def _inner_forward(x_):
                x_attn = x_ + self.ls1(self.attn(self.norm1(x_), rope=rope))
                n2 = self.norm2(x_attn)
                x_ffn = x_attn + self.ls2(self.mlp(n2, modality=modality, task=task))
                return x_ffn

            if getattr(self, "with_cp", False) and x.requires_grad:
                # torch>=2.0 推荐 use_reentrant=False；若你环境较老可删掉该参数
                try:
                    return cp.checkpoint(_inner_forward, x, use_reentrant=False)
                except TypeError:
                    return cp.checkpoint(_inner_forward, x)
            else:
                return _inner_forward(x)

        # -------------------------
        # Case 2: List[Tensor]
        # -------------------------
        elif isinstance(x_or_x_list, list):
            x_list = x_or_x_list

            # rope_list 默认补 None
            if rope_or_rope_list is None:
                rope_list = [None for _ in x_list]
            else:
                rope_list = rope_or_rope_list

            # modality 默认广播到 list
            if modality_list is None and modality is not None:
                modality_list = [modality for _ in x_list]
            if task_list is None and task is not None:
                task_list = [task for _ in x_list]

            # 这一分支是你原实现里“sample_drop_ratio > 0” 的随机子样本加速逻辑，
            # 内部有 cat/uncat/index_add，不太适合 checkpoint，保持原实现最稳 :contentReference[oaicite:1]{index=1}
            if self.training and self.sample_drop_ratio > 0.0:
                return self._forward_list(
                    x_list,
                    rope_list=rope_list,
                    modality_list=modality_list,
                    task_list=task_list,
                )

            # 否则：逐元素走正常 block，并对每个元素 checkpoint（比整体 _forward_list 更容易做）
            x_out = []
            for i, (x, rope) in enumerate(zip(x_list, rope_list)):
                m = modality_list[i] if modality_list is not None else None
                t = task_list[i] if task_list is not None else None
                def _inner_forward(x_):
                    x_attn = x_ + self.ls1(self.attn(self.norm1(x_), rope=rope))
                    n2 = self.norm2(x_attn)
                    x_ffn = x_attn + self.ls2(self.mlp(n2, modality=m, task=t))
                    return x_ffn

                if getattr(self, "with_cp", False) and x.requires_grad:
                    try:
                        x_out.append(cp.checkpoint(_inner_forward, x, use_reentrant=False))
                    except TypeError:
                        x_out.append(cp.checkpoint(_inner_forward, x))
                else:
                    x_out.append(_inner_forward(x))

            return x_out

        else:
            raise AssertionError(f"Unsupported input type: {type(x_or_x_list)}")

# -----------------------------------------------------------------------------
# DinoViT backbone
# -----------------------------------------------------------------------------
ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1)
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()


def _unwrap_sparse_checkpoint_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    for key in ("state_dict", "model", "module", "backbone", "encoder"):
        value = state_dict.get(key)
        if isinstance(value, dict):
            state_dict = value
            break

    if not isinstance(state_dict, dict):
        return state_dict

    prefixes = (
        "backbone.",
        "module.backbone.",
        "encoder.",
        "module.encoder.",
    )
    for prefix in prefixes:
        if any(k.startswith(prefix) for k in state_dict.keys()):
            state_dict = {
                k[len(prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            break
    else:
        if all(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    if "fc.weight" in state_dict or "fc.bias" in state_dict:
        state_dict = {
            k: v for k, v in state_dict.items()
            if k not in {"fc.weight", "fc.bias"}
        }

    return state_dict

@MODELS.register_module()
class DinoVisionTransformerSparseMoEFC2_LT(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        pretrained: str = 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
        arch: str = "vitb16",
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        enable_fpn: bool = False,
        device: Any | None = None,
        use_moe_fc2: bool = True,
        moe_modalities: Tuple[str, ...] = ("opt", "sar"),
        moe_num_experts: int = 4,
        moe_top_k: int = 1,
        moe_aux_loss_weight: float = 0.001,
        moe_zero_init_others: bool = False,
        moe_zero_init_router: bool = False,
        moe_alpha_init: float = 0.0001,
        moe_learnable_alpha: bool = True,
        moe_expert_type: str = "adapter",
        moe_expert_rank: int = 64,
        moe_expert_dropout: float = 0.1,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.out_channels = [embed_dim, embed_dim, embed_dim, embed_dim]
        self.pretrained = pretrained
        self.arch = arch
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))
        logger.info(f"using base={pos_embed_rope_base} for rope new")
        logger.info(f"using min_period={pos_embed_rope_min_period} for rope new")
        logger.info(f"using max_period={pos_embed_rope_max_period} for rope new")
        logger.info(f"using normalize_coords={pos_embed_rope_normalize_coords} for rope new")
        logger.info(f"using shift_coords={pos_embed_rope_shift_coords} for rope new")
        logger.info(f"using rescale_coords={pos_embed_rope_rescale_coords} for rope new")
        logger.info(f"using jitter_coords={pos_embed_rope_jitter_coords} for rope new")
        logger.info(f"using dtype={pos_embed_rope_dtype} for rope new")
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        logger.info(f"using {ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth        
        moe_cfg = dict(
            modalities=moe_modalities,
            num_experts=moe_num_experts,
            top_k=moe_top_k,
            aux_loss_weight=moe_aux_loss_weight,
            zero_init_others=moe_zero_init_others,
            zero_init_router=moe_zero_init_router,
            alpha_init=moe_alpha_init,
            learnable_alpha=moe_learnable_alpha,
            expert_type=moe_expert_type,
            expert_rank=moe_expert_rank,
            expert_dropout=moe_expert_dropout,
            num_prefix_tokens=n_storage_tokens + 1)
        
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                device=device,
                use_moe_fc2=use_moe_fc2,
                moe_cfg=moe_cfg
            )
            for i in range(depth)
        ]

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            # When untying, this norm is applied to local CLS tokens and registers.
            # This norm is never used during eval.
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))
        self.enable_fpn = enable_fpn
        self.fpn_ops = self._make_fpn_ops(embed_dim) if enable_fpn else None

    def init_weights(self):
        if hasattr(self.rope_embed, "_init_weights"):
            self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

        pretrained = getattr(self, "pretrained", None)
        if pretrained is None:
            logger.info("[init_weights] self.pretrained is None, using random initialization.")
            return
        if not isinstance(pretrained, str) or pretrained.strip() == "":
            logger.info("[init_weights] invalid pretrained=%s, using random initialization.", pretrained)
            return
        if not osp.exists(pretrained):
            raise FileNotFoundError(f"Sparse backbone checkpoint not found: {pretrained}")

        ckpt = torch.load(pretrained, map_location="cpu")
        state_dict = _unwrap_sparse_checkpoint_state_dict(ckpt)
        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(
            "[init_weights] Loaded sparse backbone from %s (loaded=%d, missing=%d, unexpected=%d)",
            pretrained,
            len(state_dict),
            len(msg.missing_keys),
            len(msg.unexpected_keys),
        )

    def gather_moe_aux_loss(self):
        """返回所有 MoE fc2 层 aux-loss 的平均；若没有则返回 0。"""
        aux = []

        for blk in self.blocks:
            if not hasattr(blk, "mlp"):
                continue
            moe_layer = None
            if hasattr(blk.mlp, "fc2") and isinstance(blk.mlp.fc2, SharedPoolSparseMoEFC2):
                moe_layer = blk.mlp.fc2
            elif hasattr(blk.mlp, "w3") and isinstance(blk.mlp.w3, SharedPoolSparseMoEFC2):
                moe_layer = blk.mlp.w3

            if moe_layer is not None:
                cur = moe_layer.get_aux_loss() if hasattr(moe_layer, "get_aux_loss") else getattr(moe_layer, "last_aux_loss", None)
                if cur is not None:
                    aux.append(cur)

        if not aux:
            return next(self.parameters()).new_tensor(0.0)

        return torch.stack(aux).mean()


    def prepare_tokens_with_masks(self, x: Tensor, masks=None) -> Tuple[Tensor, Tuple[int, ...]]:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )

        return x, (H, W)

    def forward_features_list(self, x_list, masks_list, *, modality_list=None, task_list=None):
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for _ in rope]
            x = blk(x, rope_sincos, modality_list=modality_list, task_list=task_list)
        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    x_norm_cls_reg = self.local_cls_norm(x[:, : self.n_storage_tokens + 1])
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1])
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                    "hw": rope[idx],
                }
            )
        return output

    def forward_features(self, x, masks=None, modality=None, modality_list=None):
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks], modality_list=[modality])[0]
        else:
            # list mode
            if modality_list is None and modality is not None:
                modality_list = [modality for _ in x]
            return self.forward_features_list(x, masks, modality_list=modality_list)


    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int = 1) -> List[Tensor]:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    # def forward(self, *args, is_training: bool = False, **kwargs) -> List[Dict[str, Tensor]] | Tensor:
    #     ret = self.forward_features(*args, **kwargs)
    #     if is_training:
    #         return ret
    #     if isinstance(ret, list):
    #         ret = ret[0]
    #     return self.head(ret["x_norm_clstoken"])

    def _make_fpn_ops(self, embed_dim: int) -> Optional[Tuple[nn.Module, nn.Module, nn.Module, nn.Module]]:
        """
        Build FPN-like up/down-sampling heads mirroring the ViT code you provided.
        """
        if self.patch_size == 16:
            fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                Norm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )
            fpn2 = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
            fpn3 = nn.Identity()
            fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif self.patch_size == 8:
            fpn1 = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
            fpn2 = nn.Identity()
            fpn3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
            fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=4, stride=4))
        else:
            logger.warning(f"No FPN preset for patch_size={self.patch_size}; using None.")
            return None
        return nn.ModuleList([fpn1, fpn2, fpn3, fpn4])


    def forward(
        self,
        x: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        *,
        modality: Optional[Union[str, int]] = "opt",
    ) -> Tuple[torch.Tensor, ...]:
        """
        Returns:
            Tuple of feature maps (B, C, H_i, W_i) from the four FPN branches.
        """
        if not self.enable_fpn or self.fpn_ops is None:
            raise ValueError("FPN is disabled. Recreate the model with enable_fpn=True to use this method.")

        feats = self.forward_features(
            x,
            masks=masks,
            modality=modality)

        if isinstance(feats, list):
            feats = feats[0]
        H, W = feats["hw"]
        patch_tokens = feats["x_norm_patchtokens"]  # (B, HW, C)
        B, _, C = patch_tokens.shape
        base_map = patch_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        outs = []
        for op in self.fpn_ops:
            outs.append(op(base_map))
        return tuple(outs)


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False

def mark_only_adapter_trainable(module: nn.Module) -> None:
    for name, p in module.named_parameters():
        if ".fc2." in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

def count_trainable_params(module: nn.Module) -> Tuple[int, int, float]:
    total = 0
    trainable = 0
    for p in module.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    ratio = 100.0 * trainable / max(total, 1)
    return trainable, total, ratio
# -----------------------------------------------------------------------------
# Pretrained weight helper
# -----------------------------------------------------------------------------
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "vits16": {
        "model_kwargs": dict(
            img_size=224,
            patch_size=16,
            in_chans=3,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            embed_dim=384,
            depth=12,
            num_heads=6,
            ffn_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.0,
            layerscale_init=1.0e-05,
            norm_layer="layernormbf16",
            ffn_layer="mlp",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
        ),
        "hashes": {"lvd1689m": "08c60483"},
        "weights_default": "lvd1689m",
    },
    "vits16plus": {
        "model_kwargs": dict(
            img_size=224,
            patch_size=16,
            in_chans=3,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            embed_dim=384,
            depth=12,
            num_heads=6,
            ffn_ratio=6,
            qkv_bias=True,
            drop_path_rate=0.0,
            layerscale_init=1.0e-05,
            norm_layer="layernormbf16",
            ffn_layer="swiglu",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
        ),
        "hashes": {"lvd1689m": "4057cbaa"},
        "weights_default": "lvd1689m",
    },
    "vitb16": {
        "model_kwargs": dict(
            img_size=224,
            patch_size=16,
            in_chans=3,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            embed_dim=768,
            depth=12,
            num_heads=12,
            ffn_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.0,
            layerscale_init=1.0e-05,
            norm_layer="layernormbf16",
            ffn_layer="mlp",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
        ),
        "hashes": {"lvd1689m": "73cec8be"},
        "weights_default": "lvd1689m",
    },
    "vitl16": {
        "model_kwargs": dict(
            img_size=224,
            patch_size=16,
            in_chans=3,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            embed_dim=1024,
            depth=24,
            num_heads=16,
            ffn_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.0,
            layerscale_init=1.0e-05,
            norm_layer="layernormbf16",
            ffn_layer="mlp",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
        ),
        "hashes": {"lvd1689m": "8aa4cbdd", "sat493m": "eadcf0ff"},
        "arch_overrides": {"sat493m": {"untie_global_and_local_cls_norm": True}},
        "weights_default": "lvd1689m",
    },
    "vitl16plus": {
        "model_kwargs": dict(
            img_size=224,
            patch_size=16,
            in_chans=3,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            embed_dim=1024,
            depth=24,
            num_heads=16,
            ffn_ratio=6.0,
            qkv_bias=True,
            drop_path_rate=0.0,
            layerscale_init=1.0e-05,
            norm_layer="layernormbf16",
            ffn_layer="swiglu",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
        ),
        "hashes": {"lvd1689m": "46503df0"},
        "weights_default": "lvd1689m",
    },
}


def _make_pretrained_url(arch: str, weights: str, hash_suffix: Optional[str]) -> str:
    hash_part = f"-{hash_suffix}" if hash_suffix else ""
    model_dir = f"dinov3_{arch}"
    model_filename = f"dinov3_{arch}_pretrain_{weights}{hash_part}.pth"
    return f"{DINOV3_BASE_URL}/{model_dir}/{model_filename}"


def build_dinov3_vit(
    arch: str = "vitb16",
    pretrained: bool = False,
    *,
    weights: Optional[str] = None,
    weights_path: Optional[str] = None,
    check_hash: bool = False,
    device: Optional[Union[torch.device, str]] = None,
    **override_kwargs,
) -> DinoVisionTransformer:
    """
    Build a DinoViT backbone and optionally load official pretrained weights.

    Args:
        arch: one of the keys in MODEL_CONFIGS, e.g. ``vitb16``.
        pretrained: if True, load weights from ``weights_path`` or the official URL.
        weights: weight flavor to use (defaults per arch, e.g. ``lvd1689m``).
        weights_path: local file path or URL to a state dict; overrides the default URL.
        check_hash: enable hash verification when downloading official weights.
        device: device used when constructing parameters.
        override_kwargs: forwarded to ``DinoVisionTransformer`` to tweak the config.
    """
    if arch not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported arch '{arch}'. Available: {list(MODEL_CONFIGS)}")

    cfg = MODEL_CONFIGS[arch]
    model_kwargs = dict(cfg["model_kwargs"])
    weight_key = weights or cfg.get("weights_default", "lvd1689m")
    if pretrained:
        arch_overrides = cfg.get("arch_overrides", {}).get(weight_key)
        if arch_overrides:
            model_kwargs.update(arch_overrides)
    model_kwargs.update(override_kwargs)

    model = DinoVisionTransformer(**model_kwargs, device=device)

    if pretrained:
        if weights_path is not None:
            parsed = urlparse(weights_path)
            if parsed.scheme in ("https", "http", "file"):
                state_dict = torch.hub.load_state_dict_from_url(
                    weights_path, map_location="cpu", check_hash=check_hash
                )
            else:
                state_dict = torch.load(weights_path, map_location="cpu")
        else:
            hash_suffix = cfg["hashes"].get(weight_key)
            url = _make_pretrained_url(arch=arch, weights=weight_key, hash_suffix=hash_suffix)
            state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=check_hash)
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights for {arch} with msg: {msg}")
    else:
        model.init_weights(None)

    return model


__all__ = [
    "DinoVisionTransformer",
    "build_dinov3_vit",
    "MODEL_CONFIGS",
]


def DINOV3_ViT_B_SparceMOEFC2(args):
    backbone = DinoVisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        enable_fpn=True,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k
)
    return backbone



def DINOV3_ViT_L_SparceMOEFC2_MT_LT(args):
    backbone = DinoVisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        enable_fpn=True,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_aux_loss_weight=getattr(args, "moe_aux_loss_weight", 0.001),
        moe_alpha_init=getattr(args, "moe_alpha_init", 1e-4),
        moe_expert_type=getattr(args, "moe_expert_type", "adapter"),
        moe_expert_rank=getattr(args, "moe_expert_rank", 64),
        moe_expert_dropout=getattr(args, "moe_expert_dropout", 0.1),
    )
    return backbone


def DINOV3_ViT_H_SparceMOEFC2_MT_LT(args):
    backbone = DinoVisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=6.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        enable_fpn=True,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_aux_loss_weight=getattr(args, "moe_aux_loss_weight", 0.001),
        moe_alpha_init=getattr(args, "moe_alpha_init", 1e-4),
        moe_expert_type=getattr(args, "moe_expert_type", "adapter"),
        moe_expert_rank=getattr(args, "moe_expert_rank", 64),
        moe_expert_dropout=getattr(args, "moe_expert_dropout", 0.1),
    )
    return backbone

def DINOV3_ViT_7B(args):
    backbone = DinoVisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,

        # RoPE (same as others)
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",

        # 7B core
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3,
        qkv_bias=False,

        # block details
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu64",
        ffn_bias=True,
        proj_bias=True,

        # tokens & bias
        n_storage_tokens=4,
        mask_k_bias=True,

        # IMPORTANT for 7B
        untie_global_and_local_cls_norm=True,

        # your head usage
        enable_fpn=True,
    )
    return backbone
