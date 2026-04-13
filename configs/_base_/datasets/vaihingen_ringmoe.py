# dataset settings aligned with the RingMoE / HeightFormer Vaihingen protocol
dataset_type = 'CustomDepthDataset'
data_root = 'data/vaihingen'
train_split = 'splits/vaihingen/ringmoe/train.txt'
val_split = 'splits/vaihingen/ringmoe/val.txt'
test_split = 'splits/vaihingen/ringmoe/test.txt'

raw_min_depth = 240.70
raw_max_depth = 360.00
norm_min_depth = 1e-3
norm_max_depth = 1.0

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

crop_size = (512, 512)
train_repeat_times = 64

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='DepthLoadAnnotations'),
    dict(type='RandomRotate', prob=0.5, degree=2.5),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'depth_gt'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'img_norm_cfg')),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
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

common_dataset_args = dict(
    type=dataset_type,
    data_root=data_root,
    img_dir='',
    depth_dir='',
    depth_scale=1,
    min_depth=norm_min_depth,
    max_depth=norm_max_depth,
    eval_min_depth=norm_min_depth,
    eval_max_depth=norm_max_depth,
    normalize_depth=True,
    depth_normalize_min=raw_min_depth,
    depth_normalize_max=raw_max_depth,
    depth_norm_min=norm_min_depth,
    depth_norm_max=norm_max_depth,
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=train_repeat_times,
        dataset=dict(
            **common_dataset_args,
            split=train_split,
            test_mode=False,
            pipeline=train_pipeline)),
    val=dict(
        **common_dataset_args,
        split=val_split,
        test_mode=True,
        pipeline=test_pipeline),
    test=dict(
        **common_dataset_args,
        split=test_split,
        test_mode=True,
        pipeline=test_pipeline))
