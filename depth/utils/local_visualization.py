from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .color_depth import colorize


def repo_root():
    return Path(__file__).resolve().parents[2]


def sanitize_name(name):
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)


def to_numpy(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    return value


def prepare_rgb(value):
    value = to_numpy(value)
    if value.ndim == 3 and value.shape[0] in (1, 3, 4) and value.shape[-1] not in (1, 3, 4):
        value = np.transpose(value, (1, 2, 0))
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=2)
    if value.ndim == 3 and value.shape[2] == 1:
        value = np.repeat(value, 3, axis=2)

    if value.dtype != np.uint8:
        value = value.astype(np.float32)
        if value.size > 0 and value.max() <= 1.0:
            value = value * 255.0
        value = np.clip(value, 0, 255).astype(np.uint8)

    if value.ndim == 3 and value.shape[2] == 4:
        value = value[:, :, :3]
    return value


def prepare_depth(value, cmap='magma_r', vmin=None, vmax=None):
    value = to_numpy(value).astype(np.float32)
    if value.ndim == 2:
        value = value[None, ...]
    elif value.ndim == 3 and value.shape[0] not in (1, 3, 4) and value.shape[-1] == 1:
        value = np.transpose(value, (2, 0, 1))
    if value.ndim == 3 and value.shape[0] != 1:
        value = value[:1]
    if value.ndim != 3:
        raise ValueError(f'Unsupported depth shape for visualization: {value.shape}')

    colored = colorize(value, cmap=cmap, vmin=vmin, vmax=vmax)
    if colored.ndim == 4:
        colored = colored[0]
    return colored.astype(np.uint8)


def resize_to_height(image, target_height):
    if image.shape[0] == target_height:
        return image
    pil_image = Image.fromarray(image)
    target_width = max(1, round(image.shape[1] * target_height / image.shape[0]))
    return np.array(pil_image.resize((target_width, target_height), Image.BILINEAR))


def save_visualization_triplet(output_dir,
                               prefix,
                               img_rgb=None,
                               depth_pred=None,
                               depth_gt=None,
                               depth_vmin=None,
                               depth_vmax=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared_images = {}
    if img_rgb is not None:
        prepared_images['img_rgb'] = prepare_rgb(img_rgb)
    if depth_pred is not None:
        prepared_images['img_depth_pred'] = prepare_depth(
            depth_pred, vmin=depth_vmin, vmax=depth_vmax)
    if depth_gt is not None:
        prepared_images['img_depth_gt'] = prepare_depth(
            depth_gt, vmin=depth_vmin, vmax=depth_vmax)

    safe_prefix = sanitize_name(prefix)
    for tag, image in prepared_images.items():
        Image.fromarray(image).save(output_dir / f'{safe_prefix}_{tag}.png')

    overview_tags = ['img_rgb', 'img_depth_pred', 'img_depth_gt']
    if all(tag in prepared_images for tag in overview_tags):
        target_height = max(prepared_images[tag].shape[0] for tag in overview_tags)
        overview = np.concatenate(
            [resize_to_height(prepared_images[tag], target_height) for tag in overview_tags],
            axis=1)
        Image.fromarray(overview).save(output_dir / f'{safe_prefix}_overview.png')
