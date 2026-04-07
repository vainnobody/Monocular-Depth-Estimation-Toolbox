# dataset settings
dataset_type = 'CustomDepthDataset'
train_data_root = 'data/vaihingen/train'
val_data_root = 'data/vaihingen/train'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='DepthLoadAnnotations'),
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
        img_scale=(1919, 2569),
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

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=512,
        dataset=dict(
            type=dataset_type,
            data_root=train_data_root,
            img_dir='image',
            depth_dir='dsm',
            depth_scale=1,
            test_mode=False,
            min_depth=250,
            max_depth=300,
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=val_data_root,
        img_dir='image',
        depth_dir='dsm',
        depth_scale=1,
        min_depth=250,
        max_depth=300,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=val_data_root,
        img_dir='image',
        depth_dir='dsm',
        depth_scale=1,
        min_depth=250,
        max_depth=300,
        pipeline=test_pipeline))
