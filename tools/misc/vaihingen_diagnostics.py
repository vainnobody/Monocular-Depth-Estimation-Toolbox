#!/usr/bin/env python
import argparse
import importlib.util
import json
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
HELPER_PATH = REPO_ROOT / 'depth' / 'utils' / 'vaihingen.py'


def _load_helper():
    spec = importlib.util.spec_from_file_location('vaihingen_helper', HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _none_if_empty(value):
    if value is None:
        return None
    value = value.strip()
    return value or None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inspect Vaihingen split overlap and robust depth range.')
    parser.add_argument('--data-root', default=None,
                        help='Optional dataset root. Omit when split files already use absolute image/depth paths.')
    parser.add_argument('--depth-dir', default='',
                        help='Relative depth directory used only when split second column is not absolute.')
    parser.add_argument('--train-split', required=True,
                        help='Path to train split file. Can be absolute.')
    parser.add_argument('--val-split', default=None,
                        help='Optional path to val split file. Can be absolute.')
    parser.add_argument('--default-min-depth', type=float, default=250.0)
    parser.add_argument('--default-max-depth', type=float, default=300.0)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--strict-overlap', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    helper = _load_helper()
    payload = helper.analyze_vaihingen_split_setup(
        data_root=_none_if_empty(args.data_root),
        depth_dir=args.depth_dir,
        train_split=args.train_split,
        val_split=_none_if_empty(args.val_split),
        default_min_depth=args.default_min_depth,
        default_max_depth=args.default_max_depth,
        verbose=not args.quiet,
    )
    if args.quiet:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if args.strict_overlap and payload['overlap_count'] > 0:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
