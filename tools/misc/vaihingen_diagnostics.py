import argparse
import json

from depth.utils import analyze_vaihingen_split_setup


def parse_args():
    parser = argparse.ArgumentParser(description='Inspect Vaihingen split overlap and robust depth range.')
    parser.add_argument('--data-root', default='data/vaihingen')
    parser.add_argument('--depth-dir', default='dsm')
    parser.add_argument('--train-split', default='splits/vaihingen/train.txt')
    parser.add_argument('--val-split', default='splits/vaihingen/val.txt')
    parser.add_argument('--default-min-depth', type=float, default=250.0)
    parser.add_argument('--default-max-depth', type=float, default=300.0)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--strict-overlap', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    payload = analyze_vaihingen_split_setup(
        data_root=args.data_root,
        depth_dir=args.depth_dir,
        train_split=args.train_split,
        val_split=args.val_split,
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
