import os

_base_ = ['./binsformer_dinov3b_vaihingen_24e.py']

# Set SPARSE_PRETRAINED=/abs/path/to/your_sparse_encoder.pth before launching,
# or override with:
#   --options model.backbone.pretrained=/abs/path/to/your_sparse_encoder.pth
#
# The root-level train.py hard-codes a detector checkpoint path and maps
# `encoder.* -> backbone.*`. The sparse backbone loader used here implements
# the same prefix handling for depth training and ignores unrelated decoder
# keys in that checkpoint.
sparse_pretrained = os.environ.get(
    'SPARSE_PRETRAINED',
    'pretrained/your_sparse_encoder.pth')

model = dict(
    backbone=dict(
        _delete_=True,
        type='DINOv3SparseMoEBackbone',
        model_name='base',
        arch='vitb16',
        img_size=512,
        out_indices=(2, 5, 8, 11),
        output_cls_token=True,
        pretrained=sparse_pretrained,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1e-5,
        norm_layer='layernormbf16',
        ffn_layer='mlp',
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pos_embed_rope_base=100.0,
        pos_embed_rope_normalize_coords='separate',
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype='fp32',
        use_moe_fc2=True,
        moe_modalities=('opt', 'sar'),
        moe_num_experts=4,
        moe_top_k=1,
        moe_aux_loss_weight=0.001,
        moe_zero_init_others=False,
        moe_zero_init_router=False,
        moe_alpha_init=1e-4,
        moe_learnable_alpha=True,
        moe_expert_type='adapter',
        moe_expert_rank=64,
        moe_expert_dropout=0.1,
    ))
