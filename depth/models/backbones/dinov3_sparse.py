import warnings

from mmcv.runner import BaseModule

from depth.models.builder import BACKBONES

from dinov3_sparse_moefc2_lt import DinoVisionTransformerSparseMoEFC2_LT


_MODEL_NAME_TO_ARCH = {
    'small': 'vits16',
    'base': 'vitb16',
    'large': 'vitl16',
}


@BACKBONES.register_module()
class DINOv3SparseMoEBackbone(BaseModule):
    def __init__(self,
                 model_name='base',
                 arch=None,
                 img_size=518,
                 out_indices=(2, 5, 8, 11),
                 output_cls_token=True,
                 pretrained=None,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        if arch is None:
            if model_name not in _MODEL_NAME_TO_ARCH:
                raise ValueError(
                    f'Unsupported model_name {model_name!r}. '
                    f'Choose from {tuple(_MODEL_NAME_TO_ARCH)} or pass arch explicitly.')
            arch = _MODEL_NAME_TO_ARCH[model_name]

        self.model_name = model_name
        self.arch = arch
        self.out_indices = out_indices
        self.output_cls_token = output_cls_token
        self.pretrained = pretrained

        if kwargs.get('enable_fpn', False):
            warnings.warn(
                'DINOv3SparseMoEBackbone ignores enable_fpn=True and always '
                'returns intermediate transformer features for depth heads.',
                UserWarning)
        kwargs['enable_fpn'] = False

        self.backbone = DinoVisionTransformerSparseMoEFC2_LT(
            img_size=img_size,
            pretrained=pretrained,
            arch=arch,
            **kwargs)
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size

    def init_weights(self):
        self.backbone.init_weights()

    def forward(self, inputs):
        outs = self.backbone.get_intermediate_layers(
            inputs,
            n=self.out_indices,
            reshape=True,
            return_class_token=self.output_cls_token,
            norm=True)
        return tuple(outs)
