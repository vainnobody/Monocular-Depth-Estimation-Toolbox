# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the DINOv3 License Agreement.
#
# Adapted for Monocular-Depth-Estimation-Toolbox.

import logging
import os.path as osp
import inspect
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
from mmcv.runner import BaseModule
from torch import Tensor, nn
from torch.utils import checkpoint as cp

from depth.utils import get_root_logger
from ..builder import BACKBONES
from .dinov3_layers import (
    LayerScale,
    Mlp,
    PatchEmbed,
    RMSNorm,
    RopePositionEmbedding,
    SelfAttentionBlock,
    SwiGLUFFN,
)


logger = logging.getLogger("dinov3")


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
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()


def _unwrap_checkpoint_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    for key in ("state_dict", "model", "module", "backbone"):
        value = state_dict.get(key)
        if isinstance(value, dict):
            state_dict = value
            break

    if not isinstance(state_dict, dict):
        return state_dict

    if any(k.startswith("backbone.") for k in state_dict.keys()):
        state_dict = {
            k[len("backbone."):]: v
            for k, v in state_dict.items()
            if k.startswith("backbone.")
        }
    elif any(k.startswith("module.backbone.") for k in state_dict.keys()):
        state_dict = {
            k[len("module.backbone."):]: v
            for k, v in state_dict.items()
            if k.startswith("module.backbone.")
        }
    elif all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    if "fc.weight" in state_dict or "fc.bias" in state_dict:
        state_dict = {
            k: v for k, v in state_dict.items()
            if k not in {"fc.weight", "fc.bias"}
        }

    return state_dict


class DinoVisionTransformer(BaseModule):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float = None,
        pos_embed_rope_jitter_coords: float = None,
        pos_embed_rope_rescale_coords: float = None,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        device: Any = None,
        init_cfg=None,
        **ignored_kwargs,
    ):
        super().__init__(init_cfg=init_cfg)
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

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
            self.storage_tokens = nn.Parameter(
                torch.empty(1, n_storage_tokens, embed_dim, device=device)
            )

        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            device=device,
        )

        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth

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
            )
            for i in range(depth)
        ]

        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer_cls(embed_dim)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

        self.init_weights()

    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def prepare_tokens_with_masks(
        self, x: Tensor, masks=None
    ) -> Tuple[Tensor, Tuple[int, int]]:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
            )
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

    def forward_features(
        self, x: Tensor, masks: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        x, (H, W) = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            rope_sincos = self.rope_embed(H=H, W=W)
            x = blk(x, rope_sincos)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_storage_tokens": x_norm[:, 1:self.n_storage_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.n_storage_tokens + 1:],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(
        self, x: Tensor, n: int = 1
    ) -> List[Tensor]:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len)
            if isinstance(n, int) else n
        )

        for i, blk in enumerate(self.blocks):
            rope_sincos = self.rope_embed(H=H, W=W)
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)

        assert len(output) == len(blocks_to_take), (
            f"only {len(output)} / {len(blocks_to_take)} blocks found"
        )
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)

        if norm:
            outputs = [self.norm(out) for out in outputs]

        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1:] for out in outputs]

        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]

        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training: bool = False, **kwargs) -> Tensor:
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        return self.head(ret["x_norm_clstoken"])


def vit_small(patch_size=16, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        **kwargs,
    )


def vit_base(patch_size=16, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        **kwargs,
    )


def vit_large(patch_size=16, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        **kwargs,
    )


def vit_so400m(patch_size=16, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1152,
        depth=27,
        num_heads=18,
        ffn_ratio=3.777777778,
        **kwargs,
    )


def vit_huge2(patch_size=16, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        # SwiGLU reduces the effective hidden width by 2/3 in this codebase,
        # so huge needs ratio=6 to match checkpoints with a 5120-dim FFN.
        ffn_ratio=6,
        **kwargs,
    )


def vit_giant2(patch_size=16, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        ffn_ratio=4,
        **kwargs,
    )


def build_dinov3_model(model_name, img_size=518, **kwargs):
    model_zoo = {
        "small": vit_small,
        "base": vit_base,
        "large": vit_large,
        "so400m": vit_so400m,
        "huge": vit_huge2,
        "giant": vit_giant2,
    }

    if model_name not in model_zoo:
        raise ValueError(
            f"Unknown model name: {model_name}. Available: {list(model_zoo.keys())}"
        )

    return model_zoo[model_name](
        img_size=img_size,
        patch_size=16,
        ffn_layer="mlp" if model_name not in ["giant", "huge", "so400m"] else "swiglu",
        n_storage_tokens=4,
        mask_k_bias=True,
        layerscale_init=1.0e-05,
        pos_embed_rope_rescale_coords=2,
        norm_layer="layernormbf16",
        **kwargs,
    )


@BACKBONES.register_module()
class DINOv3Backbone(BaseModule):
    def __init__(self,
                 model_name='base',
                 img_size=518,
                 out_indices=(2, 5, 8, 11),
                 output_cls_token=True,
                 with_cp=False,
                 pretrained='pretrained/dinov3_base.pth',
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.model_name = model_name
        self.out_indices = out_indices
        self.output_cls_token = output_cls_token
        self.with_cp = with_cp
        self.pretrained = pretrained
        self.backbone = build_dinov3_model(model_name=model_name, img_size=img_size)
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size

    def init_weights(self):
        if not self.pretrained:
            return

        if not osp.exists(self.pretrained):
            raise FileNotFoundError(
                f'DINOv3 checkpoint not found: {self.pretrained}. '
                'Please place the backbone weights at this path.')

        logger_ = get_root_logger()
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = _unwrap_checkpoint_state_dict(checkpoint)
        load_result = self.backbone.load_state_dict(state_dict, strict=False)
        logger_.info(
            'Loaded DINOv3 backbone from %s (missing=%d, unexpected=%d)',
            self.pretrained,
            len(load_result.missing_keys),
            len(load_result.unexpected_keys))

    def _forward_block(self, blk, block_idx, x, rope):
        # Do not checkpoint blocks whose outputs are tapped as multi-scale features:
        # reentrant backward on branched outputs can trip DDP's "mark ready twice".
        use_cp = (
            self.with_cp and self.training and x.requires_grad
            and block_idx not in self.out_indices
        )
        if use_cp:
            sin, cos = rope
            checkpoint_kwargs = {}
            if 'use_reentrant' in inspect.signature(cp.checkpoint).parameters:
                checkpoint_kwargs['use_reentrant'] = False
            return cp.checkpoint(
                lambda inp, sin_inp, cos_inp: blk(inp, (sin_inp, cos_inp)),
                x,
                sin,
                cos,
                **checkpoint_kwargs)
        return blk(x, rope)

    def forward(self, inputs):
        x, (H, W) = self.backbone.prepare_tokens_with_masks(inputs)
        output, blocks_to_take = [], set(self.out_indices)

        for i, blk in enumerate(self.backbone.blocks):
            rope_sincos = self.backbone.rope_embed(H=H, W=W)
            x = self._forward_block(blk, i, x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)

        assert len(output) == len(self.out_indices), (
            f'only {len(output)} / {len(self.out_indices)} blocks found')

        output = [self.backbone.norm(out) for out in output]
        class_tokens = [out[:, 0] for out in output]
        output = [out[:, self.backbone.n_storage_tokens + 1:] for out in output]

        B, _, h, w = inputs.shape
        output = [
            out.reshape(B, h // self.patch_size, w // self.patch_size, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
            for out in output
        ]

        if self.output_cls_token:
            outs = tuple(zip(output, class_tokens))
        else:
            outs = tuple(output)
        return tuple(outs)
