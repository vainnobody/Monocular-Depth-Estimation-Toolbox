_base_ = [
    '../_base_/models/binsformer.py',
    '../_base_/datasets/vaihingen.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_24x.py',
]

norm_cfg = dict(type='BN', requires_grad=True)
manual_min_depth = 1e-3
manual_max_depth = 500.0
manual_eval_min_depth = 1e-3
manual_eval_max_depth = 500.0

model = dict(
    backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3, 4),
        style='pytorch',
        norm_cfg=norm_cfg,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrained/resnet50.pth')),
    decode_head=dict(
        type='BinsFormerDecodeHead',
        classify=False,
        in_channels=[256, 512, 1024, 2048],
        conv_dim=256,
        min_depth=manual_min_depth,
        max_depth=manual_max_depth,
        n_bins=64,
        index=[0, 1, 2, 3],
        trans_index=[1, 2, 3],
        loss_decode=dict(type='SigLoss', valid_mask=True, loss_weight=10),
        with_loss_chamfer=False,
        norm_cfg=norm_cfg,
        transformer_encoder=dict(
            type='PureMSDEnTransformer',
            num_feature_levels=3,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=3,
                        num_points=8),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        transformer_decoder=dict(
            type='PixelTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            num_feature_levels=3,
            hidden_dim=256,
            transformerlayers=dict(
                type='PixelTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0),
                ffn_cfgs=dict(
                    feedforward_channels=2048,
                    ffn_drop=0.0),
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')))),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)))

data = dict(
    train=dict(
        dataset=dict(
            min_depth=manual_min_depth,
            max_depth=manual_max_depth,
            eval_min_depth=manual_eval_min_depth,
            eval_max_depth=manual_eval_max_depth)),
    val=dict(
        min_depth=manual_min_depth,
        max_depth=manual_max_depth,
        eval_min_depth=manual_eval_min_depth,
        eval_max_depth=manual_eval_max_depth),
    test=dict(
        min_depth=manual_min_depth,
        max_depth=manual_max_depth,
        eval_min_depth=manual_eval_min_depth,
        eval_max_depth=manual_eval_max_depth))

find_unused_parameters = True

max_lr = 1e-4
optimizer = dict(
    type='AdamW',
    lr=max_lr,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=1.0),
            'decode_head': dict(lr_mult=10.0),
        }))

lr_config = dict(
    policy='OneCycle',
    max_lr=max_lr,
    div_factor=25,
    final_div_factor=100,
    by_epoch=False)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
evaluation = dict(
    by_epoch=True,
    interval=1,
    pre_eval=True,
    save_viz=True,
    viz_dir='viz',
    rule='less',
    save_best='abs_rel',
    greater_keys=('a1', 'a2', 'a3'),
    less_keys=('abs_rel', 'rmse'))

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(
            type='LocalVisualizationHook',
            by_epoch=True,
            out_dir='viz',
            interval=50,
            ignore_last=False),
    ])
