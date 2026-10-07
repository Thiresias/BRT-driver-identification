"""Boundary, decode-failure and filtered-manifest regression tests."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

AVAILABLE = all(importlib.util.find_spec(name) for name in ('numpy', 'cv2'))
if AVAILABLE:
    import cv2
    import numpy as np
    from tools.metadata import VideoRecord, load_metadata, group_records
    from tools.video_sampling import (SamplingConfig, VideoInfo, VideoError, plan_window,
                                      decode_window, select_record)
    from tools.validate_videos import validate_videos


class LastRng:
    def randint(self, low, high=None):
        return (low if high is None else high) - 1


@unittest.skipUnless(AVAILABLE, 'requires numpy and OpenCV')
class SamplingTests(unittest.TestCase):
    def test_exact_minimum_arbitrary_fps(self):
        for fps in (1, 15, 23.976, 24, 29.97, 60, 120):
            with self.subTest(fps=fps):
                window = plan_window(VideoInfo(40, fps), LastRng(), SamplingConfig())
                self.assertEqual(window.start, 0)
                self.assertEqual(window.offsets, tuple(range(40)))

    def test_last_start_and_max_stride_are_inclusive(self):
        window = plan_window(VideoInfo(210, 60), LastRng(), SamplingConfig())
        self.assertEqual(window.offsets, tuple(range(0, 196, 5)))
        self.assertEqual(window.start, 14)
        self.assertEqual(window.start + window.offsets[-1], 209)

    def test_short_and_invalid_metadata_fail_before_rng(self):
        for count, fps in [(39, 30), (0, 30), (-1, 24), (40, 0), (40, float('nan'))]:
            with self.subTest(count=count, fps=fps), self.assertRaises(VideoError):
                plan_window(VideoInfo(count, fps), None, SamplingConfig())

    def test_fixed_mode_general_fps(self):
        for fps in (15, 23.976, 25, 29.97, 30, 60):
            window = plan_window(VideoInfo(1000, fps), LastRng(), SamplingConfig('fixed'))
            self.assertEqual(len(set(window.offsets)), 40)
            self.assertLess(window.start + window.offsets[-1], 1000)
        with self.assertRaises(VideoError):
            plan_window(VideoInfo(1000, 2), None, SamplingConfig('fixed'))

    def test_errors_identify_clip_and_do_not_reference_unassigned_start(self):
        record = VideoRecord('trump:12345', Path('/video/broken.mp4'), 'genuine', 'trump', 'trump', '1', 'train')
        with self.assertRaisesRegex(VideoError, r'trump:12345.*broken.mp4.*frames=39'):
            decode_window(record, VideoInfo(39, 60), None, SamplingConfig())
        with patch('tools.video_sampling.cv2.VideoCapture') as capture:
            cap = capture.return_value
            cap.isOpened.return_value = True
            cap.set.return_value = True
            cap.read.return_value = (False, None)
            with self.assertRaisesRegex(VideoError, 'decoding failed at frame 0'):
                decode_window(record, VideoInfo(40, 60), LastRng(), SamplingConfig())
            cap.release.assert_called_once()


@unittest.skipUnless(AVAILABLE, 'requires numpy and OpenCV')
class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = self.root / 'metadata.json'
        self.output = self.root / 'filtered' / 'metadata.csv'

    def video(self, name, count, fps=24):
        path = self.root / (name + '.avi')
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (256, 256))
        self.assertTrue(writer.isOpened())
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        for _ in range(count):
            writer.write(frame)
        writer.release()
        return path

    def row(self, name, path, driver='trump', recording=None):
        return dict(clip_id=name, path=str(path), kind='genuine', driver_id=driver,
                    appearance_id=driver, driver_video_id=recording or name, split='train')

    def save(self, rows):
        self.manifest.write_text(json.dumps(rows))

    def test_filter_short_corrupt_and_keep_arbitrary_fps(self):
        rows = [self.row('p15', self.video('p15', 40, 15)),
                self.row('p23976', self.video('p23976', 80, 23.976)),
                self.row('negative', self.video('negative', 80, 60), 'other'),
                self.row('short', self.video('short', 39))]
        broken = self.root / 'broken.avi'
        broken.write_bytes(b'not a video')
        rows.append(self.row('broken', broken))
        self.save(rows)
        original = self.manifest.read_bytes()
        summary = validate_videos([self.manifest], self.output, 'trump', full=True)
        self.assertTrue(summary['usable'])
        self.assertEqual(summary['accepted_clips'], 3)
        self.assertEqual(summary['rejected_clips'], 2)
        self.assertEqual(self.manifest.read_bytes(), original)
        records = load_metadata(self.output)
        self.assertTrue(all(r.path.is_file() for r in records))
        self.assertEqual({r.clip_id for r in records}, {'p15', 'p23976', 'negative'})
        self.assertIn('short', self.output.with_name('metadata.rejected.csv').read_text())
        if importlib.util.find_spec('torch'):
            from LIA_encoder.metadata_dataset import ManifestVideoDataset
            dataset = ManifestVideoDataset(records, 'trump', 'train')
            for index in range(len(dataset)):
                self.assertEqual(tuple(dataset[index][0].shape), (10, 40, 3, 256, 256))

    def test_class_loss_is_reported_and_not_usable(self):
        self.save([self.row('short', self.video('short', 39)),
                   self.row('negative', self.video('negative', 40), 'other')])
        summary = validate_videos([self.manifest], self.output, 'trump')
        self.assertFalse(summary['usable'])
        self.assertTrue(summary['issues'])
        with self.assertRaisesRegex(ValueError, 'both POI'):
            group_records(load_metadata(self.output), 'train', 'trump')

    def test_generator_validation_and_direct_validator_cli(self):
        repository = Path(__file__).resolve().parents[1]
        for role, fps in [('trump', 15), ('world', 60)]:
            source = self.video(role, 40, fps)
            destination = self.root / role / 'train' / '001.avi'
            destination.parent.mkdir(parents=True)
            source.rename(destination)
        result = subprocess.run(
            [sys.executable, 'create_metadata.py', '--root', str(self.root), '--poi', 'trump',
             '--validate-videos'], cwd=repository / 'tools', text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / 'trump' / 'metadata.validated.csv').is_file())
        result = subprocess.run(
            [sys.executable, 'validate_videos.py', '--metadata', str(self.root / 'trump' / 'metadata.csv'),
             '--poi', 'trump', '--output', str(self.output)],
            cwd=repository / 'tools', text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)['usable'])

    def test_input_csv_cannot_be_overwritten(self):
        import csv
        rows = [self.row('p', self.video('p', 40)), self.row('n', self.video('n', 40), 'other')]
        source = self.root / 'source.csv'
        with source.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, 'overwrite input'):
            validate_videos([source], source, 'trump', force=True)

    def test_revalidate_groups_after_filtering(self):
        rows = [self.row('good', self.root / 'good.avi', recording='same'),
                self.row('bad', self.root / 'bad.avi', recording='same'),
                self.row('negative', self.root / 'negative.avi', driver='other')]
        self.save(rows)
        calls = []
        def decode(record, info, rng, config):
            calls.append(record.clip_id)
            if len(calls) > 30 and record.clip_id == 'bad':
                raise VideoError('simulated group-window decode failure')
        with patch('tools.validate_videos.probe_video', return_value=VideoInfo(100, 60)), \
             patch('tools.validate_videos.decode_window', side_effect=decode):
            result = validate_videos([self.manifest], self.output, 'trump')
        self.assertTrue(result['usable'])
        self.assertEqual(result['rejected_clips'], 1)
        self.assertEqual(set(calls[-20:]), {'good', 'negative'})
        self.assertEqual({r.clip_id for r in load_metadata(self.output)}, {'good', 'negative'})
