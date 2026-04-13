#!/usr/bin/env python3
"""Re-evaluate saved predictions under the normalized Vaihingen protocol."""

import argparse
import pickle
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare raw-domain and normalized-domain metrics on Vaihingen predictions.')
    parser.add_argument('predictions', help='Path to the pickle file saved by tools/test.py --out.')
    parser.add_argument(
        '--data-root',
        default='data/vaihingen',
        help='Dataset root used to resolve split entries.')
    parser.add_argument(
        '--split',
        default='splits/vaihingen/ringmoe/test.txt',
        help='Split file used to align predictions and ground-truth labels.')
    parser.add_argument(
        '--pred-domain',
        choices=['raw', 'normalized'],
        default='raw',
        help='Whether predictions are still in raw DSM meters or already normalized to [0, 1].')
    parser.add_argument('--raw-min-depth', type=float, default=240.70)
    parser.add_argument('--raw-max-depth', type=float, default=360.00)
    parser.add_argument('--norm-min-depth', type=float, default=1e-3)
    parser.add_argument('--norm-max-depth', type=float, default=1.0)
    parser.add_argument(
        '--compare-raw-domain',
        action='store_true',
        help='Also compute metrics in the raw DSM domain for side-by-side comparison.')
    return parser.parse_args()


def read_split_entries(path: Path):
    entries = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        image_rel, depth_rel = line.split()[:2]
        entries.append((image_rel, depth_rel))
    return entries


def load_predictions(path: Path):
    with path.open('rb') as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, list):
        raise TypeError(f'Expected a list of predictions, got {type(payload)!r}.')
    if payload and isinstance(payload[0], tuple):
        raise TypeError(
            'The pickle file contains pre-eval tuples instead of depth maps. '
            'Re-run tools/test.py with --out only, without --eval.')
    return payload


def squeeze_prediction(prediction):
    array = np.asarray(prediction, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f'Expected a 2D depth map, got shape {array.shape}.')
    return array


def normalize_depth_map(depth_map, src_min, src_max, dst_min, dst_max):
    depth_map = np.asarray(depth_map, dtype=np.float32)
    normalized = depth_map.copy()
    valid_mask = np.isfinite(depth_map) & (depth_map > 0)
    if not np.any(valid_mask):
        return normalized
    scale = (dst_max - dst_min) / max(src_max - src_min, 1e-12)
    clipped = np.clip(depth_map[valid_mask], src_min, src_max)
    normalized[valid_mask] = (clipped - src_min) * scale + dst_min
    return normalized


def calculate_metrics(gt, pred):
    thresh = np.maximum(gt / pred, pred / gt)
    log_diff = np.log(pred) - np.log(gt)
    silog_var = max(float(np.mean(log_diff ** 2) - np.mean(log_diff) ** 2), 0.0)
    return {
        'a1': float((thresh < 1.25).mean()),
        'a2': float((thresh < 1.25 ** 2).mean()),
        'a3': float((thresh < 1.25 ** 3).mean()),
        'abs_rel': float(np.mean(np.abs(gt - pred) / gt)),
        'rmse': float(np.sqrt(np.mean((gt - pred) ** 2))),
        'rmse_log': float(np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))),
        'log_10': float(np.mean(np.abs(np.log10(gt) - np.log10(pred)))),
        'silog': float(np.sqrt(silog_var) * 100),
        'sq_rel': float(np.mean(((gt - pred) ** 2) / gt)),
    }


def evaluate_pair(gt_map, pred_map, min_depth, max_depth):
    mask = np.isfinite(gt_map) & np.isfinite(pred_map)
    mask &= gt_map > min_depth
    mask &= gt_map < max_depth
    mask &= pred_map > 0
    if not np.any(mask):
        raise ValueError('No valid pixels remained after masking.')
    return calculate_metrics(gt_map[mask], pred_map[mask])


def average_metrics(metrics_per_image):
    keys = metrics_per_image[0].keys()
    return {key: float(np.mean([metrics[key] for metrics in metrics_per_image])) for key in keys}


def print_metrics(title, metrics):
    print(title)
    for key in ('abs_rel', 'a1', 'a2', 'a3', 'rmse', 'rmse_log', 'log_10', 'silog', 'sq_rel'):
        print(f'  {key}: {metrics[key]:.6f}')


def main():
    args = parse_args()
    predictions = load_predictions(Path(args.predictions))
    split_entries = read_split_entries(Path(args.split))
    if len(predictions) != len(split_entries):
        raise ValueError(
            f'Prediction count ({len(predictions)}) does not match split size ({len(split_entries)}).')

    data_root = Path(args.data_root)
    raw_metrics = []
    normalized_metrics = []

    for prediction, (_, depth_rel) in zip(predictions, split_entries):
        gt_raw = np.asarray(Image.open(data_root / depth_rel), dtype=np.float32)
        pred_map = squeeze_prediction(prediction)
        if pred_map.shape != gt_raw.shape:
            raise ValueError(
                f'Prediction shape {pred_map.shape} does not match GT shape {gt_raw.shape} for {depth_rel}.')

        gt_normalized = normalize_depth_map(
            gt_raw,
            src_min=args.raw_min_depth,
            src_max=args.raw_max_depth,
            dst_min=args.norm_min_depth,
            dst_max=args.norm_max_depth)
        if args.pred_domain == 'raw':
            pred_normalized = normalize_depth_map(
                pred_map,
                src_min=args.raw_min_depth,
                src_max=args.raw_max_depth,
                dst_min=args.norm_min_depth,
                dst_max=args.norm_max_depth)
        else:
            pred_normalized = pred_map

        normalized_metrics.append(
            evaluate_pair(
                gt_normalized,
                pred_normalized,
                min_depth=args.norm_min_depth,
                max_depth=args.norm_max_depth))

        if args.compare_raw_domain:
            pred_raw = pred_map
            if args.pred_domain == 'normalized':
                scale = (args.raw_max_depth - args.raw_min_depth) / max(
                    args.norm_max_depth - args.norm_min_depth, 1e-12)
                pred_raw = (pred_map - args.norm_min_depth) * scale + args.raw_min_depth
            raw_metrics.append(
                evaluate_pair(
                    gt_raw,
                    pred_raw,
                    min_depth=args.raw_min_depth,
                    max_depth=args.raw_max_depth))

    print_metrics('Normalized protocol metrics', average_metrics(normalized_metrics))
    if args.compare_raw_domain:
        print_metrics('Raw DSM protocol metrics', average_metrics(raw_metrics))


if __name__ == '__main__':
    main()
