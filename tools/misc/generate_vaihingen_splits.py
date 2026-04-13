#!/usr/bin/env python3
"""Generate the official 16/17 Vaihingen split with flexible path layouts."""

import argparse
from pathlib import Path


TRAIN_TILE_IDS = [1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32, 34, 37]
TEST_TILE_IDS = [2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29, 31, 33, 35, 38]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate RingMoE-style Vaihingen train/val/test split files.')
    parser.add_argument(
        '--data-root',
        default='data/vaihingen',
        help='Dataset root to scan recursively.')
    parser.add_argument(
        '--output-dir',
        default='splits/vaihingen/ringmoe',
        help='Directory used to write train/val/test split files.')
    parser.add_argument(
        '--image-token',
        default='image',
        help='Directory token used to identify RGB images while scanning.')
    parser.add_argument(
        '--depth-token',
        default='dsm',
        help='Directory token used to identify depth maps while scanning.')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the resolved paths without writing output files.')
    return parser.parse_args()


def tile_name(tile_id):
    return f'top_mosaic_09cm_area{tile_id}.tif'


def find_unique_file(root: Path, filename: str, required_token: str) -> Path:
    matches = []
    for candidate in root.rglob(filename):
        if required_token in candidate.parts:
            matches.append(candidate)
    if not matches:
        raise FileNotFoundError(
            f'Could not find "{filename}" under "{root}" with token "{required_token}".')
    if len(matches) > 1:
        raise RuntimeError(
            f'Found multiple matches for "{filename}" with token "{required_token}": '
            + ', '.join(str(path) for path in matches))
    return matches[0]


def build_entries(root: Path, tile_ids, image_token: str, depth_token: str):
    entries = []
    for tile_id in tile_ids:
        filename = tile_name(tile_id)
        image_path = find_unique_file(root, filename, image_token)
        depth_path = find_unique_file(root, filename, depth_token)
        entries.append((
            image_path.relative_to(root).as_posix(),
            depth_path.relative_to(root).as_posix(),
        ))
    return entries


def write_split_file(path: Path, header: str, entries):
    lines = [header]
    lines.extend(f'{image_path} {depth_path}' for image_path, depth_path in entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir)

    train_entries = build_entries(root, TRAIN_TILE_IDS, args.image_token, args.depth_token)
    test_entries = build_entries(root, TEST_TILE_IDS, args.image_token, args.depth_token)

    if args.dry_run:
        print('[train]')
        for image_path, depth_path in train_entries:
            print(image_path, depth_path)
        print('[test]')
        for image_path, depth_path in test_entries:
            print(image_path, depth_path)
        return

    write_split_file(
        output_dir / 'train.txt',
        '# RingMoE / HeightFormer-style official Vaihingen train split.',
        train_entries)
    write_split_file(
        output_dir / 'val.txt',
        '# RingMoE / HeightFormer-style validation split mirrored from test.',
        test_entries)
    write_split_file(
        output_dir / 'test.txt',
        '# RingMoE / HeightFormer-style official Vaihingen test split.',
        test_entries)

    print(f'Wrote {len(train_entries)} train and {len(test_entries)} test entries to {output_dir}.')


if __name__ == '__main__':
    main()
