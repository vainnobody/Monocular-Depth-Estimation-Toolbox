_base_ = [
    '../_base_/models/adabins.py', '../_base_/datasets/vaihingen.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_24x.py'
]

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    decode_head=dict(
        min_depth=250,
        max_depth=300,
        norm_cfg=norm_cfg),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)))

find_unused_parameters = True
SyncBN = True

# optimizer
max_lr = 1e-4
optimizer = dict(
    type='AdamW',
    lr=max_lr,
    weight_decay=0.1,
    paramwise_cfg=dict(custom_keys={'decode_head': dict(lr_mult=10)}))

# learning policy
lr_config = dict(
    policy='OneCycle',
    max_lr=max_lr,
    div_factor=25,
    final_div_factor=100,
    by_epoch=False,
)

momentum_config = dict(policy='OneCycle')

# runtime
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
evaluation = dict(interval=1)
