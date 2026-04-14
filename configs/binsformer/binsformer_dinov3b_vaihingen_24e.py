_base_ = [
    '../_base_/models/binsformer.py',
    '../_base_/datasets/vaihingen.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_24x.py',
]

norm_cfg = dict(type='BN', requires_grad=True)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
# RingMoE/HeightFormer-style Vaihingen protocol:
# normalize raw DSM heights from dataset range into [1e-3, 1.0] before
# training / validation, then evaluate Rel and delta metrics in normalized
# space to avoid absolute terrain offsets inflating a1/a2/a3.
raw_min_depth = 240.70
raw_max_depth = 360.00
norm_min_depth = 1e-3
norm_max_depth = 1.0

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=True,
        flip_direction='horizontal',
        transforms=[
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor', 'flip',
                           'flip_direction', 'img_norm_cfg')),
        ])
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='DINOv3Backbone',
        model_name='base',
        img_size=512,
        out_indices=(2, 5, 8, 11),
        output_cls_token=True,
        pretrained='pretrained/dinov3_base.pth'),
    neck=dict(
        type='DINOv3AdaBinsNeck',
        in_channels=768,
        out_channels=[96, 192, 384, 768],
        readout_type='project',
        patch_size=16),
    decode_head=dict(
        type='BinsFormerDecodeHead',
        classify=False,
        in_channels=[96, 192, 384, 768],
        conv_dim=256,
        min_depth=norm_min_depth,
        max_depth=norm_max_depth,
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
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(256, 256)))

data = dict(
    train=dict(
        dataset=dict(
            min_depth=norm_min_depth,
            max_depth=norm_max_depth,
            eval_min_depth=norm_min_depth,
            eval_max_depth=norm_max_depth,
            normalize_depth=True,
            depth_normalize_min=raw_min_depth,
            depth_normalize_max=raw_max_depth,
            depth_norm_min=norm_min_depth,
            depth_norm_max=norm_max_depth)),
    val=dict(
        min_depth=norm_min_depth,
        max_depth=norm_max_depth,
        eval_min_depth=norm_min_depth,
        eval_max_depth=norm_max_depth,
        pipeline=test_pipeline,
        normalize_depth=True,
        depth_normalize_min=raw_min_depth,
        depth_normalize_max=raw_max_depth,
        depth_norm_min=norm_min_depth,
        depth_norm_max=norm_max_depth),
    test=dict(
        min_depth=norm_min_depth,
        max_depth=norm_max_depth,
        eval_min_depth=norm_min_depth,
        eval_max_depth=norm_max_depth,
        pipeline=test_pipeline,
        normalize_depth=True,
        depth_normalize_min=raw_min_depth,
        depth_normalize_max=raw_max_depth,
        depth_norm_min=norm_min_depth,
        depth_norm_max=norm_max_depth))

find_unused_parameters = True

# Keep the Vaihingen normalization protocol above, but align the optimizer
# and runner with the official BinsFormer toolbox recipe as closely as
# possible for this custom DINOv3 backbone variant.
max_lr = 1e-4
optimizer = dict(
    type='AdamW',
    lr=max_lr,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.),
        }))

lr_config = dict(
    policy='OneCycle',
    max_lr=max_lr,
    warmup_iters=1600 * 8,
    div_factor=25,
    final_div_factor=100,
    by_epoch=False)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
runner = dict(_delete_=True, type='IterBasedRunner', max_iters=1600 * 24)
checkpoint_config = dict(_delete_=True, by_epoch=False, max_keep_ckpts=2, interval=1600)
evaluation = dict(
    _delete_=True,
    by_epoch=False,
    start=0,
    interval=1600,
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
        dict(type='TextLoggerHook', by_epoch=False),
        dict(
            type='LocalVisualizationHook',
            by_epoch=False,
            out_dir='viz',
            interval=50,
            ignore_last=False),
    ])
