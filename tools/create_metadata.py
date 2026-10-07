"""Create genuine/driver-identification CSV manifests without decoding videos."""

import argparse
import csv
import hashlib
import math
import os
import re
import tempfile
from pathlib import Path

if __package__:
    from .metadata import load_metadata
else:
    # Direct execution (including from inside tools/) uses the sibling module.
    from metadata import load_metadata

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
SPLITS = {'train': 'train', 'val': 'validation', 'validation': 'validation', 'test': 'test'}
FIELDS = ['clip_id', 'path', 'kind', 'driver_scope', 'driver_id', 'poi_id',
          'appearance_id', 'driver_video_id', 'split', 'start_sec', 'end_sec',
          'source_asset_id', 'method']
TIMES = re.compile(r'\[\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*\]$')
PART = re.compile(r'^(?P<video>.+)_part_\d+$', re.IGNORECASE)


def video_paths(folder):
    return sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def filename_metadata(path):
    """Parse the documented video_part_N [start - end] convention only."""
    stem = path.stem.strip()
    match = TIMES.search(stem)
    start = end = ''
    if match:
        start, end = map(float, match.groups())
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError(f'{path}: invalid timestamp interval')
        stem = stem[:match.start()].strip()
    elif '[' in stem or ']' in stem:
        raise ValueError(f'{path}: malformed timestamp suffix; expected [start - end]')
    part = PART.fullmatch(stem)
    return (part.group('video') if part else stem), start, end


def scan_folder(folder, poi, world, output, val_fraction=None, seed=0):
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f'input directory does not exist: {folder}')
    paths = video_paths(folder)
    if not paths:
        raise ValueError(f'no video files found in {folder}')
    rows, flat = [], []
    has_presplit = any((folder / split).is_dir() for split in ('train', 'val', 'validation'))
    if has_presplit and val_fraction is not None:
        raise ValueError(f'{folder}: do not use --val-fraction with existing train/val splits')
    if (folder / 'val').exists() and (folder / 'validation').exists():
        raise ValueError(f'{folder}: use either val or validation, not both')
    role = 'world' if world else poi
    for path in paths:
        relative = path.relative_to(folder)
        split = SPLITS.get(relative.parts[0]) if len(relative.parts) > 1 else None
        if split is None and (has_presplit or val_fraction is None):
            raise ValueError(f'{path}: expected train/val/test subfolders; use --val-fraction for unsplit genuine videos')
        within = Path(*relative.parts[1:]) if split else relative
        kind = 'generated' if split == 'test' else 'genuine'
        recording, start, end = filename_metadata(path)
        # Filenames of generated assets cannot establish their original driver recording.
        recording_id = (role + '/' + (within.parent / recording).as_posix()) if kind == 'genuine' else ''
        clip_key = role + '/' + relative.as_posix()
        row = dict(clip_id=clip_key,
                   path=os.path.relpath(path, output.parent), kind=kind,
                   driver_scope='non_poi' if world else 'named',
                   driver_id='' if world else poi, poi_id=poi if world else '',
                   appearance_id=poi if kind == 'genuine' and not world else '',
                   driver_video_id=recording_id, split=split or 'train',
                   start_sec=start,
                   end_sec=end, source_asset_id='', method='')
        rows.append(row)
        if split is None:
            flat.append(row)
    if val_fraction is not None:
        if not 0 < val_fraction < 1:
            raise ValueError('--val-fraction must be between 0 and 1')
        recordings = sorted({row['driver_video_id'] for row in flat},
                            key=lambda key: hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest())
        if len(recordings) < 2:
            raise ValueError(f'{folder}: need at least two original genuine recordings to split')
        count = max(1, min(len(recordings) - 1, round(len(recordings) * val_fraction)))
        validation = set(recordings[:count])
        for row in flat:
            row['split'] = 'validation' if row['driver_video_id'] in validation else 'train'
    return rows


def assign_short_ids(rows):
    """Allocate role-prefixed five-hex IDs, resolving collisions within this CSV."""
    capacity = 16 ** 5
    if len(rows) > capacity:
        raise ValueError(f'five-digit clip IDs support at most {capacity} videos per CSV')
    used = set()
    for row in sorted(rows, key=lambda item: item['clip_id']):
        candidate = int(hashlib.sha256(row['clip_id'].encode()).hexdigest()[:5], 16)
        while candidate in used:
            candidate = (candidate + 1) % capacity
        used.add(candidate)
        prefix = 'world' if row['driver_scope'] == 'non_poi' else row['driver_id']
        row['clip_id'] = f'{prefix}:{candidate:05x}'


def output_path(output, poi_dir, world_dir):
    if output is None:
        folder = poi_dir or world_dir
        if folder is None:
            raise ValueError('provide --poi-dir and/or --world-dir')
        output = Path(folder) / 'metadata.csv'
    return Path(output).expanduser().resolve()


def create_metadata(output, poi, poi_dir=None, world_dir=None, val_fraction=None, seed=0, force=False):
    if not poi.strip() or poi.strip() != poi or poi.lower() == 'world':
        raise ValueError('--poi must name the actual POI, not world, with no surrounding whitespace')
    if not poi_dir and not world_dir:
        raise ValueError('provide --poi-dir and/or --world-dir')
    if poi_dir and world_dir:
        a, b = Path(poi_dir).resolve(), Path(world_dir).resolve()
        if a == b or a in b.parents or b in a.parents:
            raise ValueError('POI and world input directories must not overlap')
    output = output_path(output, poi_dir, world_dir)
    if output.suffix.lower() != '.csv':
        raise ValueError('--output must end in .csv')
    if output.exists() and not force:
        raise ValueError(f'{output} exists; use --force to replace it')
    rows = []
    for folder, world in [(poi_dir, False), (world_dir, True)]:
        if folder:
            rows.extend(scan_folder(folder, poi, world, output, val_fraction, seed))
    assign_short_ids(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate before publishing; failed validation leaves an existing output intact.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                         suffix='.csv', dir=output.parent, delete=False) as handle:
            temp_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        load_metadata(temp_path)
        if force:
            os.replace(temp_path, output)
        else:
            # Exclusive creation avoids overwriting a file created after our initial check.
            os.link(temp_path, output)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, help='Dataset root containing <poi>/ and optionally world/')
    parser.add_argument('--poi-dir', type=Path, help='POI folder (alternative to --root)')
    parser.add_argument('--world-dir', type=Path, help='Known non-POI folder (alternative to --root)')
    parser.add_argument('--poi', required=True, help='Actual person of interest; world is not an identity')
    parser.add_argument('--output', type=Path, help='Default: metadata.csv inside the POI folder (world folder for world-only input)')
    parser.add_argument('--val-fraction', type=float, help='Explicitly split unsplit genuine recordings')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--force', action='store_true', help='Replace an existing CSV after validation')
    parser.add_argument('--validate-videos', action='store_true', help='Also produce a filtered manifest and rejection report')
    parser.add_argument('--full-decode', action='store_true', help='With --validate-videos, also decode every frame')
    parser.add_argument('--sampling', choices=['multiscale', 'fixed'], default='multiscale')
    parser.add_argument('--max-stride', type=int, default=5)
    args = parser.parse_args()
    if args.full_decode and not args.validate_videos:
        parser.error('--full-decode requires --validate-videos')
    if args.root:
        if args.poi_dir or args.world_dir:
            parser.error('--root cannot be combined with --poi-dir/--world-dir')
        args.poi_dir = args.root / args.poi
        args.world_dir = args.root / 'world' if (args.root / 'world').is_dir() else None
    try:
        args.output = output_path(args.output, args.poi_dir, args.world_dir)
        rows = create_metadata(args.output, args.poi, args.poi_dir, args.world_dir,
                               args.val_fraction, args.seed, args.force)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f'Wrote {len(rows)} videos to {args.output}')
    for split in ('train', 'validation', 'test'):
        selected = [row for row in rows if row['split'] == split]
        if selected:
            positives = sum(row['driver_scope'] == 'named' for row in selected)
            print(f'{split}: {positives} POI clips, {len(selected) - positives} world clips')
    if any(row['driver_scope'] == 'non_poi' for row in rows):
        print(f'world is an explicit assertion that all its drivers are not {args.poi}; identities remain unknown.')
    print('Genuine recording IDs follow the documented filename convention and relative subfolders; review them before training.')
    if args.validate_videos:
        if __package__:
            from .validate_videos import validate_videos
        else:
            from validate_videos import validate_videos
        filtered = args.output.with_name(args.output.stem + '.validated.csv')
        try:
            summary = validate_videos([args.output], filtered, args.poi, args.sampling,
                                      args.max_stride, args.full_decode, args.force)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(f"Validated {summary['accepted_clips']} clips; rejected {summary['rejected_clips']}. See {filtered}")
        if not summary['usable']:
            parser.exit(2, 'Filtered data is not usable: see the validation report.\n')



if __name__ == '__main__':
    main()
