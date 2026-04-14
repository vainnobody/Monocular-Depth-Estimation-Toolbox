from depth.models import depther
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from depth.core import add_prefix
from depth.ops import resize
from depth.models import builder
from depth.models.builder import DEPTHER
from .base import BaseDepther

# for model size
import numpy as np

@DEPTHER.register_module()
class DepthEncoderDecoder(BaseDepther):
    """Encoder Decoder depther.

    EncoderDecoder typically consists of backbone, (neck) and decode_head.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(DepthEncoderDecoder, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and depther set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        self._init_decode_head(decode_head)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas, rescale=True):
        """Encode images with backbone and decode into a depth estimation
        map of the same size as input."""
        
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        # crop the pred depth to the certain range.
        out = torch.clamp(out, min=self.decode_head.min_depth, max=self.decode_head.max_depth)
        if rescale:
            out = resize(
                input=out,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, img, x, img_metas, depth_gt, **kwargs):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(img, x, img_metas, depth_gt, self.train_cfg, **kwargs)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        depth_pred = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return depth_pred

    def forward_dummy(self, img):
        """Dummy forward function."""
        depth = self.encode_decode(img, None)

        return depth

    def forward_train(self, img, img_metas, depth_gt, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `depth/datasets/pipelines/formatting.py:Collect`.
            depth_gt (Tensor): Depth gt
                used if the architecture supports depth estimation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)

        losses = dict()

        # the last of x saves the info from neck
        loss_decode = self._decode_head_forward_train(img, x, img_metas, depth_gt, **kwargs)
 
        losses.update(loss_decode)

        return losses

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        depth_pred = self.encode_decode(img, img_meta, rescale)

        return depth_pred

    def _build_slide_blend_weight(self, crop_h, crop_w, device, dtype):
        """Create a smooth weighting mask for overlap-add slide inference.

        Equal averaging keeps seams when crop-border predictions are less
        stable. A Hann-style center-weighted mask suppresses those borders
        while still covering the full crop.
        """
        if crop_h <= 1:
            weight_h = torch.ones((crop_h,), device=device, dtype=dtype)
        else:
            weight_h = torch.hann_window(
                crop_h, periodic=False, device=device, dtype=dtype)
        if crop_w <= 1:
            weight_w = torch.ones((crop_w,), device=device, dtype=dtype)
        else:
            weight_w = torch.hann_window(
                crop_w, periodic=False, device=device, dtype=dtype)

        weight = weight_h[:, None] * weight_w[None, :]
        # Avoid zeros on crop borders so image boundaries that are only covered
        # once remain numerically stable.
        weight = weight.clamp_min(1e-3)
        return weight[None, None, :, :]

    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap."""
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()

        preds = img.new_zeros((batch_size, 1, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                crop_img = img[:, :, y1:y2, x1:x2]
                crop_img_metas = copy.deepcopy(img_meta)
                for crop_img_meta in crop_img_metas:
                    crop_shape = crop_img.shape[2:] + (crop_img.shape[1], )
                    crop_img_meta['img_shape'] = crop_shape
                    crop_img_meta['pad_shape'] = crop_shape

                crop_depth_pred = self.encode_decode(
                    crop_img, crop_img_metas, rescale=True)
                crop_weight = self._build_slide_blend_weight(
                    crop_depth_pred.shape[2],
                    crop_depth_pred.shape[3],
                    device=crop_depth_pred.device,
                    dtype=crop_depth_pred.dtype)

                preds += F.pad(
                    crop_depth_pred * crop_weight,
                    (
                        int(x1),
                        int(preds.shape[3] - x2),
                        int(y1),
                        int(preds.shape[2] - y2),
                    ))
                count_mat += F.pad(
                    crop_weight.expand(batch_size, -1, -1, -1),
                    (
                        int(x1),
                        int(preds.shape[3] - x2),
                        int(y1),
                        int(preds.shape[2] - y2),
                    ))

        assert (count_mat == 0).sum() == 0
        depth_pred = preds / count_mat

        if rescale:
            ori_shape = img_meta[0]['ori_shape']
            depth_pred = resize(
                input=depth_pred,
                size=ori_shape[:2],
                mode='bilinear',
                align_corners=self.align_corners)

        return depth_pred

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `depth/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output depth map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            depth_pred = self.slide_inference(img, img_meta, rescale)
        else:
            depth_pred = self.whole_inference(img, img_meta, rescale)
        output = depth_pred
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        depth_pred = self.inference(img, img_meta, rescale)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            depth_pred = depth_pred.unsqueeze(0)
            return depth_pred
        depth_pred = depth_pred.cpu().numpy()
        # unravel batch dim
        depth_pred = list(depth_pred)
        return depth_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented depth logit inplace
        depth_pred = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_depth_pred = self.inference(imgs[i], img_metas[i], rescale)
            depth_pred += cur_depth_pred
        depth_pred /= len(imgs)
        depth_pred = depth_pred.cpu().numpy()
        # unravel batch dim
        depth_pred = list(depth_pred)
        return depth_pred
