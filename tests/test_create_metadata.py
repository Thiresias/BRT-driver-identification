import csv
import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch
from pathlib import Path

from tools.metadata import group_records, load_metadata
from tools.create_metadata import create_metadata, filename_metadata, assign_short_ids


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'manifests' / 'metadata.csv'

    def video(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def generate(self, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return create_metadata(self.output, 'trump', **kwargs)

    def load(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return load_metadata(self.output)

    def layout(self):
        for role in ['trump', 'world']:
            for split, filename in [('train', '151_part_1 [11.40 - 29.76].mp4'),
                                    ('val', '160.mp4'), ('test', 'fake.mp4')]:
                self.video(f'{role}/{split}/{filename}')

    def test_presplit_roundtrip_labels_paths_and_unknowns(self):
        self.layout()
        self.video('trump/train/151_part_1 [57.38 - 75.70].mp4')
        rows = self.generate(poi_dir=self.root / 'trump', world_dir=self.root / 'world')
        records = self.load()
        self.assertEqual(len(rows), 7)
        for split in ['train', 'validation', 'test']:
            groups = group_records(records, split, 'trump')
            self.assertEqual(len(groups), 2)
            self.assertEqual({g.label for g in groups}, {0, 1})
        train = [r for r in records if r.split == 'train' and r.driver_id == 'trump']
        self.assertEqual(len({r.driver_video_id for r in train}), 1)
        self.assertEqual(len({r.clip_id for r in train}), 2)
        self.assertTrue(all(r.path.is_file() for r in records))
        self.assertTrue(all(r.driver_id == '' for r in records if r.driver_scope == 'non_poi'))
        self.assertTrue(all(r.driver_video_id == '' and r.appearance_id == ''
                            for r in records if r.kind == 'generated'))
        self.assertTrue(all(r.start_sec is None for r in records if r.split == 'validation'))
        with self.assertRaisesRegex(ValueError, 'scoped to POI'):
            group_records(records, 'train', 'obama')

    def test_generated_files_remain_separate(self):
        self.layout()
        self.video('trump/test/fake_part_2.mp4')
        self.generate(poi_dir=self.root / 'trump', world_dir=self.root / 'world')
        self.assertEqual(len(group_records(self.load(), 'test', 'trump')), 3)

    def test_missing_and_nan_timestamps(self):
        self.video('trump/train/001.mp4')
        self.generate(poi_dir=self.root / 'trump')
        with self.output.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual((rows[0]['start_sec'], rows[0]['end_sec']), ('', ''))
        for missing in [('NaN', 'nan'), (None, None)]:
            rows[0]['start_sec'], rows[0]['end_sec'] = missing
            path = self.root / 'test.json'
            path.write_text(json.dumps(rows))
            self.assertIsNone(load_metadata(path)[0].start_sec)

    def test_split_whole_recordings_reproducibly(self):
        for role in ['trump', 'world']:
            for recording in range(5):
                for part in [1, 2]:
                    self.video(f'{role}/{recording}_part_{part}.mp4')
            self.video(f'{role}/test/fake.mp4')
        options = dict(poi_dir=self.root / 'trump', world_dir=self.root / 'world', val_fraction=.4, seed=42)
        rows = self.generate(**options)
        self.assertEqual(rows, self.generate(**options, force=True))
        groups = {}
        for row in rows:
            if row['kind'] == 'genuine':
                groups.setdefault(row['driver_video_id'], set()).add(row['split'])
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        self.assertEqual(sum(splits == {'validation'} for splits in groups.values()), 4)
        records = self.load()
        for split in ['train', 'validation', 'test']:
            self.assertEqual({g.label for g in group_records(records, split, 'trump')}, {0, 1})

    def test_no_implicit_split(self):
        self.video('trump/001.mp4')
        with self.assertRaisesRegex(ValueError, 'val-fraction'):
            self.generate(poi_dir=self.root / 'trump')

    def test_existing_split_collision_rejected_and_output_preserved(self):
        self.video('trump/train/151_part_1.mp4')
        self.video('trump/val/151_part_2.mp4')
        self.output.parent.mkdir()
        self.output.write_text('keep me')
        with self.assertRaisesRegex(ValueError, 'appears in both'):
            self.generate(poi_dir=self.root / 'trump', force=True)
        self.assertEqual(self.output.read_text(), 'keep me')

    def test_world_relative_subfolders_namespace_recordings(self):
        self.video('world/train/person_a/001_part_1.mp4')
        self.video('world/train/person_b/001_part_1.mp4')
        self.generate(world_dir=self.root / 'world')
        self.assertEqual(len({r.recording_key for r in self.load()}), 2)

    def test_validation_alias_and_nonvideo_files(self):
        self.video('trump/validation/001.MP4')
        self.video('trump/validation/notes.txt')
        rows = self.generate(poi_dir=self.root / 'trump')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['split'], 'validation')

    def test_fail_on_ambiguous_layout_overwrite_and_bad_times(self):
        self.layout()
        self.generate(poi_dir=self.root / 'trump')
        with self.assertRaisesRegex(ValueError, 'exists'):
            self.generate(poi_dir=self.root / 'trump')
        with self.assertRaisesRegex(ValueError, 'existing train/val'):
            self.generate(poi_dir=self.root / 'trump', force=True, val_fraction=.2)
        self.video('trump/stray.mp4')
        with self.assertRaisesRegex(ValueError, 'expected train/val/test'):
            self.generate(poi_dir=self.root / 'trump', force=True)
        for name in ['001 [bad].mp4', '001 [10 - 1].mp4']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                filename_metadata(Path(name))

    def test_root_cli(self):
        self.layout()
        result = subprocess.run([sys.executable, '-m', 'tools.create_metadata',
                                 '--root', str(self.root), '--poi', 'trump', '--output', str(self.output)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Wrote 6 videos', result.stdout)
        self.assertIn('recording-overlap checks are incomplete', result.stderr)

    def test_cli_from_repository_and_tools_directory(self):
        self.layout()
        repository = Path(__file__).resolve().parents[1]
        invocations = [
            (repository, ['-m', 'tools.create_metadata'], ['-m', 'tools.metadata']),
            (repository, ['tools/create_metadata.py'], ['tools/metadata.py']),
            (repository / 'tools', ['create_metadata.py'], ['metadata.py']),
            (repository / 'tools', ['-m', 'create_metadata'], ['-m', 'metadata']),
        ]
        for cwd, generator, validator in invocations:
            with self.subTest(cwd=str(cwd), generator=generator):
                result = subprocess.run(
                    [sys.executable, *generator, '--root', str(self.root), '--poi', 'trump',
                     '--output', str(self.output), '--force'], cwd=cwd, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                result = subprocess.run(
                    [sys.executable, *validator, '--metadata', str(self.output), '--poi', 'trump'],
                    cwd=cwd, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                summary = json.loads(result.stdout)
                for split in ('train', 'validation', 'test'):
                    self.assertEqual(summary[split]['positive_groups'], 1)
                    self.assertEqual(summary[split]['negative_groups'], 1)

    def test_short_ids_are_unique_and_repeatable(self):
        self.layout()
        options = dict(poi_dir=self.root / 'trump', world_dir=self.root / 'world')
        rows = self.generate(**options)
        ids = [row['clip_id'] for row in rows]
        for row in rows:
            prefix, suffix = row['clip_id'].rsplit(':', 1)
            self.assertEqual(prefix, 'world' if row['driver_scope'] == 'non_poi' else 'trump')
            self.assertRegex(suffix, r'^[0-9a-f]{5}$')
        self.assertEqual(len(set(ids)), len(rows))
        self.assertEqual(ids, [row['clip_id'] for row in self.generate(**options, force=True)])

    def test_short_id_collisions_wrap_without_duplicates(self):
        rows = [dict(clip_id=key, driver_scope='named', driver_id='trump') for key in ['b', 'a', 'c']]
        with patch('tools.create_metadata.hashlib.sha256') as digest:
            digest.return_value.hexdigest.return_value = 'fffff' + '0' * 59
            assign_short_ids(rows)
        self.assertEqual([row['clip_id'] for row in rows], ['trump:00000', 'trump:fffff', 'trump:00001'])

    def test_default_output_in_poi_folder_and_relative_media_paths(self):
        self.layout()
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, '-m', 'tools.create_metadata', '--root', str(self.root), '--poi', 'trump'],
            cwd=repository, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = self.root / 'trump' / 'metadata.csv'
        self.assertTrue(output.is_file())
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            records = load_metadata(output)
        self.assertTrue(all(record.path.is_file() for record in records))
        self.assertEqual(len(records), 6)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            create_metadata(None, 'trump', world_dir=self.root / 'world')
        self.assertTrue((self.root / 'world' / 'metadata.csv').is_file())

    def test_partial_scope_and_unknown_recording_validation(self):
        rows = [dict(clip_id='a', path='a.mp4', kind='generated', split='test',
                     driver_id='trump'),
                dict(clip_id='b', path='b.mp4', kind='generated', split='test',
                     driver_scope='non_poi', poi_id='trump')]
        path = self.root / 'manifest.json'
        path.write_text(json.dumps(rows))
        with self.assertWarnsRegex(UserWarning, 'incomplete'):
            records = load_metadata(path)
        self.assertEqual({g.label for g in group_records(records, 'test', 'trump')}, {0, 1})
        for changes in [dict(poi_id=''), dict(driver_id='obama'), dict(driver_scope='anything')]:
            rows[1] = dict(clip_id='b', path='b.mp4', kind='generated', split='test',
                           driver_scope='non_poi', poi_id='trump')
            rows[1].update(changes)
            path.write_text(json.dumps(rows))
            with self.assertRaises(ValueError):
                load_metadata(path)


if __name__ == '__main__':
    unittest.main()
