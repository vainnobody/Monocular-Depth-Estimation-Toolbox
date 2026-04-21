# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the DINOv3 License Agreement.

from typing import Callable, List, Optional

import torch.nn.functional as F
from torch import Tensor, nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        device=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, device=device)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
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
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # DINOv3 checkpoints store SwiGLU gate widths directly at the FFN
        # hidden size (e.g. ViT-H+ uses 5120), so do not shrink by 2/3 here.
        swiglu_hidden_features = hidden_features + (-hidden_features % align_to)
        self.w1 = nn.Linear(
            in_features, swiglu_hidden_features, bias=bias, device=device
        )
        self.w2 = nn.Linear(
            in_features, swiglu_hidden_features, bias=bias, device=device
        )
        self.w3 = nn.Linear(
            swiglu_hidden_features, out_features, bias=bias, device=device
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)
