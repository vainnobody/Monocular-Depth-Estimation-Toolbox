_base_ = ['./binsformer_dinov3b_vaihingen_24e.py']

# Replace this with your sparse/MoE encoder checkpoint, or override with:
#   --options model.backbone.pretrained=/abs/path/to/your_sparse_encoder.pth
#
# The sparse backbone loader supports checkpoints whose backbone weights are
# stored under `encoder.*`, `module.encoder.*`, `backbone.*`, or
# `module.backbone.*`, and ignores unrelated decoder keys.
sparse_pretrained = 'pretrained/your_sparse_encoder.pth'

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
