import os

_base_ = ['./binsformer_dinov3b_vaihingen_24e.py']


model = dict(
    backbone=dict(
        _delete_=True,
        type='DINOv3Backbone',
        model_name='huge',
        img_size=512,
        # DINOv3 huge has 32 transformer blocks; sample four evenly spaced stages.
        out_indices=(7, 15, 23, 31),
        output_cls_token=True,
        with_cp=True,
        pretrained='pretrained/dinov3_huge.pth'),
    neck=dict(
        type='DINOv3AdaBinsNeck',
        in_channels=1280,
        out_channels=[160, 320, 640, 1280],
        readout_type='project',
        patch_size=16),
    decode_head=dict(
        type='BinsFormerDecodeHead',
        in_channels=[160, 320, 640, 1280]))

# DINOv3 huge is substantially heavier than base; default to a safer batch size
# while still allowing server-side overrides via the existing env var.
data = dict(samples_per_gpu=int(os.getenv('VAIHINGEN_SAMPLES_PER_GPU', '1')))

find_unused_parameters = False

del os
