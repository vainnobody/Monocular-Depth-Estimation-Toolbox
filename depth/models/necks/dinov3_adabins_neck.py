# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Linear, build_activation_layer
from mmcv.runner import BaseModule

from depth.models.builder import NECKS


class ReassembleBlocks(BaseModule):
    def __init__(self,
                 in_channels=768,
                 out_channels=[96, 192, 384, 768],
                 readout_type='project',
                 patch_size=16,
                 init_cfg=None):
        super().__init__(init_cfg)

        assert readout_type in ['ignore', 'add', 'project']
        self.readout_type = readout_type
        self.patch_size = patch_size

        self.projects = nn.ModuleList([
            ConvModule(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                act_cfg=None,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        if self.readout_type == 'project':
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        Linear(2 * in_channels, in_channels),
                        build_activation_layer(dict(type='GELU'))))

    def forward(self, inputs):
        assert isinstance(inputs, (list, tuple))
        out = []
        for i, x in enumerate(inputs):
            assert len(x) == 2
            x, cls_token = x[0], x[1]
            feature_shape = x.shape
            if self.readout_type == 'project':
                x = x.flatten(2).permute((0, 2, 1))
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
                x = x.permute(0, 2, 1).reshape(feature_shape)
            elif self.readout_type == 'add':
                x = x.flatten(2) + cls_token.unsqueeze(-1)
                x = x.reshape(feature_shape)

            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        return tuple(out)


@NECKS.register_module()
class DINOv3AdaBinsNeck(BaseModule):
    def __init__(self,
                 in_channels=768,
                 out_channels=[96, 192, 384, 768],
                 readout_type='project',
                 patch_size=16,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.reassemble_blocks = ReassembleBlocks(
            in_channels=in_channels,
            out_channels=out_channels,
            readout_type=readout_type,
            patch_size=patch_size)

    def forward(self, inputs):
        return self.reassemble_blocks(inputs)
