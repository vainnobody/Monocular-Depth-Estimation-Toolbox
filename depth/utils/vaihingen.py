from __future__ import annotations

import json
import math
import os
import os.path as osp
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SplitSignature:
    path: str
    size: int
    mtime: float


def _resolve_path(data_root: Optional[str], path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if data_root is not None and not osp.isabs(path):
        return osp.join(data_root, path)
    return path


def _resolve_split_path(data_root: Optional[str], split: Optional[str]) -> Optional[str]:
    if split is None:
        return None
    if osp.isabs(split):
        return split
    candidate = osp.join(data_root, split) if data_root is not None else split
    if osp.exists(candidate):
        return candidate
    return split


def _read_split_entries(split_path: str) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    with open(split_path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            fields = line.split()
            if len(fields) == 1:
                entries.append((fields[0], fields[0]))
            else:
                entries.append((fields[0], fields[1]))
    return entries


def _split_signature(path: Optional[str]) -> Optional[SplitSignature]:
    if path is None or not osp.exists(path):
        return None
    stat = os.stat(path)
    return SplitSignature(path=path, size=stat.st_size, mtime=stat.st_mtime)


def _default_cache_path(data_root: Optional[str]) -> Optional[str]:
    if data_root is None:
        return None
    return osp.join(data_root, '.vaihingen_depth_stats.json')


def _load_cache(cache_path: Optional[str]) -> Optional[dict]:
    if cache_path is None or not osp.exists(cache_path):
        return None
    try:
        with open(cache_path) as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(cache_path: Optional[str], payload: dict) -> None:
    if cache_path is None:
        return
    try:
        os.makedirs(osp.dirname(cache_path), exist_ok=True)
        tmp_path = cache_path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp_path, cache_path)
    except Exception:
        pass


def analyze_vaihingen_split_setup(
    *,
    data_root: Optional[str],
    depth_dir: str,
    train_split: str,
    val_split: Optional[str] = None,
    valid_min_depth: float = 0.0,
    lower_percentile: float = 0.1,
    upper_percentile: float = 99.9,
    default_min_depth: float = 250.0,
    default_max_depth: float = 300.0,
    cache_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    resolved_depth_dir = _resolve_path(data_root, depth_dir)
    resolved_train_split = _resolve_split_path(data_root, train_split)
    resolved_val_split = _resolve_split_path(data_root, val_split)
    cache_path = cache_path or _default_cache_path(data_root)

    cache_key = {
        'data_root': data_root,
        'depth_dir': resolved_depth_dir,
        'train_split': vars(_split_signature(resolved_train_split)) if _split_signature(resolved_train_split) else None,
        'val_split': vars(_split_signature(resolved_val_split)) if _split_signature(resolved_val_split) else None,
        'valid_min_depth': valid_min_depth,
        'lower_percentile': lower_percentile,
        'upper_percentile': upper_percentile,
    }

    cached = _load_cache(cache_path)
    if cached and cached.get('cache_key') == cache_key:
        payload = cached['payload']
        if verbose:
            print('[Vaihingen] loaded cached diagnostics from', cache_path)
            print('[Vaihingen] split overlap:', payload['overlap_count'])
            print('[Vaihingen] recommended depth range: [{}, {}]'.format(
                payload['recommended_min_depth'], payload['recommended_max_depth']))
        return payload

    payload: Dict[str, object] = {
        'resolved_train_split': resolved_train_split,
        'resolved_val_split': resolved_val_split,
        'resolved_depth_dir': resolved_depth_dir,
        'overlap_count': 0,
        'overlap_examples': [],
        'train_count': 0,
        'val_count': 0,
        'num_train_depth_maps': 0,
        'num_valid_pixels': 0,
        'dataset_min_depth': float(default_min_depth),
        'dataset_max_depth': float(default_max_depth),
        'p0_1_depth': float(default_min_depth),
        'p1_depth': float(default_min_depth),
        'p50_depth': float((default_min_depth + default_max_depth) / 2.0),
        'p99_depth': float(default_max_depth),
        'p99_9_depth': float(default_max_depth),
        'recommended_min_depth': float(default_min_depth),
        'recommended_max_depth': float(default_max_depth),
        'used_defaults': True,
        'missing_depth_maps': [],
    }

    if not resolved_train_split or not osp.exists(resolved_train_split):
        if verbose:
            print('[Vaihingen] train split not found, using default depth range [{}, {}]'.format(
                default_min_depth, default_max_depth))
        return payload

    train_entries = _read_split_entries(resolved_train_split)
    val_entries = _read_split_entries(resolved_val_split) if resolved_val_split and osp.exists(resolved_val_split) else []
    payload['train_count'] = len(train_entries)
    payload['val_count'] = len(val_entries)

    train_images = {img_name for img_name, _ in train_entries}
    val_images = {img_name for img_name, _ in val_entries}
    overlap = sorted(train_images & val_images)
    payload['overlap_count'] = len(overlap)
    payload['overlap_examples'] = overlap[:20]

    depth_values: List[np.ndarray] = []
    min_depth = math.inf
    max_depth = -math.inf
    total_valid_pixels = 0
    missing_depth_maps: List[str] = []

    for _, depth_name in train_entries:
        depth_path = depth_name if osp.isabs(depth_name) else osp.join(resolved_depth_dir, depth_name)
        if not osp.exists(depth_path):
            missing_depth_maps.append(depth_path)
            continue
        depth = np.asarray(Image.open(depth_path), dtype=np.float32)
        valid_mask = np.isfinite(depth) & (depth > valid_min_depth)
        if not np.any(valid_mask):
            continue
        valid_depth = depth[valid_mask].reshape(-1)
        depth_values.append(valid_depth)
        total_valid_pixels += int(valid_depth.size)
        min_depth = min(min_depth, float(valid_depth.min()))
        max_depth = max(max_depth, float(valid_depth.max()))

    payload['missing_depth_maps'] = missing_depth_maps[:20]
    payload['num_train_depth_maps'] = len(train_entries) - len(missing_depth_maps)
    payload['num_valid_pixels'] = total_valid_pixels

    if depth_values:
        merged = np.concatenate(depth_values, axis=0)
        q0_1, q1, q50, q99, q99_9 = np.percentile(
            merged, [lower_percentile, 1.0, 50.0, 99.0, upper_percentile])
        rec_min = float(math.floor(q0_1))
        rec_max = float(math.ceil(q99_9))
        if rec_max <= rec_min:
            rec_min = float(math.floor(min_depth))
            rec_max = float(math.ceil(max_depth))
        payload.update({
            'dataset_min_depth': float(min_depth),
            'dataset_max_depth': float(max_depth),
            'p0_1_depth': float(q0_1),
            'p1_depth': float(q1),
            'p50_depth': float(q50),
            'p99_depth': float(q99),
            'p99_9_depth': float(q99_9),
            'recommended_min_depth': rec_min,
            'recommended_max_depth': rec_max,
            'used_defaults': False,
        })

    if verbose:
        print('[Vaihingen] train samples={}, val samples={}, overlap={}'.format(
            payload['train_count'], payload['val_count'], payload['overlap_count']))
        if overlap:
            print('[Vaihingen] WARNING: overlapping split entries detected, examples={}'.format(payload['overlap_examples']))
        if missing_depth_maps:
            print('[Vaihingen] WARNING: missing depth maps encountered, examples={}'.format(payload['missing_depth_maps']))
        print('[Vaihingen] valid train pixels={}'.format(payload['num_valid_pixels']))
        print('[Vaihingen] depth stats min={:.3f}, p0.1={:.3f}, p1={:.3f}, p50={:.3f}, p99={:.3f}, p99.9={:.3f}, max={:.3f}'.format(
            payload['dataset_min_depth'], payload['p0_1_depth'], payload['p1_depth'], payload['p50_depth'], payload['p99_depth'], payload['p99_9_depth'], payload['dataset_max_depth']))
        print('[Vaihingen] recommended depth range: [{}, {}]{}'.format(
            int(payload['recommended_min_depth']), int(payload['recommended_max_depth']),
            ' (default)' if payload['used_defaults'] else ''))

    _save_cache(cache_path, {'cache_key': cache_key, 'payload': payload})
    return payload
