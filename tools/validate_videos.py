"""Filter unusable video clips before training; preserve input manifests and media."""

import argparse
import csv
import json
import os
import tempfile
from dataclasses import asdict, fields
from pathlib import Path

if __package__:
    from .metadata import VideoRecord, group_records, load_metadata
    from .video_sampling import (NUM_WINDOWS, SamplingConfig, VideoError, decode_window,
                                 full_decode, probe_video, select_record)
else:
    from metadata import VideoRecord, group_records, load_metadata
    from video_sampling import (NUM_WINDOWS, SamplingConfig, VideoError, decode_window,
                                full_decode, probe_video, select_record)


def write_csv(path, names, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def validate_videos(manifests, output, poi, sampling='multiscale', max_stride=5,
                    full=False, force=False):
    manifests = [Path(path).resolve() for path in manifests]
    output = Path(output).resolve()
    if output.suffix.lower() != '.csv':
        raise ValueError('output must end in .csv')
    rejected_path = output.with_name(output.stem + '.rejected.csv')
    summary_path = output.with_name(output.stem + '.validation.json')
    targets = [output, rejected_path, summary_path]
    records = load_metadata(manifests)
    protected = set(manifests) | {record.path.resolve() for record in records}
    for target in targets:
        if target in protected:
            raise ValueError(f'output must not overwrite input metadata/media: {target}')
        if target.exists() and not force:
            raise ValueError(f'{target} exists; use --force to replace validation outputs')
    config = SamplingConfig(sampling, max_stride)
    splits = sorted({record.split for record in records})
    # Fail on metadata/POI mistakes rather than treating them as damaged videos.
    for split in splits:
        group_records(records, split, poi)

    rejected, info_by_path = {}, {}
    for record in records:
        try:
            info = probe_video(record, config)
            info_by_path[record.path] = info
            if full:
                full_decode(record, info)
            # Even clips not selected by the current group draw get decode coverage.
            for window in range(NUM_WINDOWS):
                _, rng = select_record((record,), window)
                decode_window(record, info, rng, config)
        except VideoError as exc:
            rejected[record.clip_id] = dict(clip_id=record.clip_id, path=str(record.path),
                                           split=record.split, stage='clip', reason=str(exc))

    # Removing clips changes group selection. Repeat until all windows in the
    # FINAL filtered groups have been decoded with the exact dataset RNG draws.
    while True:
        accepted = [record for record in records if record.clip_id not in rejected]
        failures = {}
        for split in splits:
            try:
                groups = group_records(accepted, split, poi)
            except ValueError:
                continue  # Report missing classes/empty splits below.
            for group in groups:
                for window in range(NUM_WINDOWS):
                    record, rng = select_record(group.records, window)
                    try:
                        decode_window(record, info_by_path[record.path], rng, config)
                    except VideoError as exc:
                        failures[record.clip_id] = dict(clip_id=record.clip_id, path=str(record.path),
                                                       split=record.split, stage='group', reason=str(exc))
        if not failures:
            break
        rejected.update(failures)

    split_summary, issues = {}, []
    for split in splits:
        selected = [record for record in accepted if record.split == split]
        split_summary[split] = dict(clips=len(selected),
                                   positive_clips=sum(r.label_for(poi) for r in selected),
                                   negative_clips=sum(1 - r.label_for(poi) for r in selected))
        try:
            groups = group_records(accepted, split, poi)
            split_summary[split]['groups'] = len(groups)
        except ValueError as exc:
            issues.append(str(exc))
    summary = dict(usable=not issues, issues=issues, input_clips=len(records),
                   accepted_clips=len(accepted), rejected_clips=len(rejected),
                   poi=poi, sampling=asdict(config), windows=NUM_WINDOWS,
                   full_decode=full, splits=split_summary,
                   inputs=[str(path) for path in manifests], output=str(output),
                   rejections=str(rejected_path),
                   note='Sample validation covers deterministic windows, not all frames. '
                        'Files must remain unchanged after validation; decoding does not detect all visual corruption.')
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in accepted:
        row = asdict(record)
        row['path'] = os.path.relpath(record.path, output.parent)
        rows.append(row)
    # Finish all output serialization before replacing any validation output.
    with tempfile.TemporaryDirectory(dir=output.parent) as folder:
        folder = Path(folder)
        write_csv(folder / 'accepted.csv', [field.name for field in fields(VideoRecord)], rows)
        write_csv(folder / 'rejected.csv', ['clip_id', 'path', 'split', 'stage', 'reason'],
                  [rejected[key] for key in sorted(rejected)])
        (folder / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        for staged, target in zip(['accepted.csv', 'rejected.csv', 'summary.json'], targets):
            if force:
                os.replace(folder / staged, target)
            else:
                os.link(folder / staged, target)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--poi', required=True)
    parser.add_argument('--sampling', choices=['multiscale', 'fixed'], default='multiscale')
    parser.add_argument('--max-stride', type=int, default=5)
    parser.add_argument('--full-decode', action='store_true', help='Also decode every frame sequentially')
    parser.add_argument('--force', action='store_true', help='Replace existing validation outputs')
    args = parser.parse_args()
    try:
        summary = validate_videos(args.metadata, args.output, args.poi, args.sampling,
                                  args.max_stride, args.full_decode, args.force)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))
    if not summary['usable']:
        parser.exit(2, 'Filtered data is not usable: see issues in the validation report.\n')


if __name__ == '__main__':
    main()
