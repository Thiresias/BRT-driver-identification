"""CPU integration checks with synthetic videos; no research data/weights needed."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "cv2", "numpy"))
if AVAILABLE:
    import cv2
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from LIA_encoder.metadata import load_metadata
    from LIA_encoder.metadata_dataset import ManifestVideoDataset


@unittest.skipUnless(AVAILABLE, "install torch, numpy and opencv-python-headless for video integration tests")
class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rows = []
        for speaker, fps, color in [("trump", 25, (0, 0, 255)), ("obama", 30, (255, 0, 0))]:
            folder = self.root / speaker
            folder.mkdir()
            path = folder / "000.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (256, 256))
            self.assertTrue(writer.isOpened(), "MJPG encoder unavailable")
            for _ in range(8 * fps):
                writer.write(np.full((256, 256, 3), color, dtype=np.uint8))
            writer.release()
            rows.append(dict(clip_id=speaker, path=str(path), kind="genuine", driver_id=speaker,
                             appearance_id=speaker, driver_video_id="000", split="test"))
        manifest = self.root / "metadata.json"
        manifest.write_text(json.dumps(rows))
        self.records = load_metadata(manifest)

    def test_tensor_contract_and_default_collation(self):
        dataset = ManifestVideoDataset(self.records, "trump", "test")
        frames, labels, ids, fps = next(iter(DataLoader(dataset, batch_size=2)))
        self.assertEqual(tuple(frames.shape), (2, 10, 40, 3, 256, 256))
        self.assertEqual(labels.tolist(), [[0] * 10, [1] * 10])
        self.assertEqual(list(ids[0]), dataset.video_ids)
        self.assertTrue(all(ids[k] == ids[0] for k in range(10)))
        self.assertEqual(fps[:, 0].tolist(), [30, 25])
        # The 25 FPS Trump fixture is red in RGB, checking channel conversion.
        self.assertGreater(frames[1, 0, 0, 0].float().mean().item(), 240)
        self.assertLess(frames[1, 0, 0, 2].float().mean().item(), 10)

    def test_repeatable_windows_do_not_reset_global_rng(self):
        dataset = ManifestVideoDataset(self.records, "trump", "test")
        np.random.seed(912)
        expected = np.random.RandomState(912).random_sample()
        first = dataset[0]
        self.assertEqual(np.random.random_sample(), expected)
        second = dataset[0]
        self.assertTrue(torch.equal(first[0], second[0]))

    def test_generated_poi_world_csv_loads_real_videos(self):
        from tools.create_metadata import create_metadata
        for record in self.records:
            role = 'trump' if record.driver_id == 'trump' else 'world'
            destination = self.root / role / 'test' / 'fake.avi'
            destination.parent.mkdir(parents=True, exist_ok=True)
            record.path.rename(destination)
        manifest = self.root / 'generated.csv'
        with self.assertWarns(UserWarning):
            create_metadata(manifest, 'trump', self.root / 'trump', self.root / 'world')
        with self.assertWarns(UserWarning):
            records = load_metadata(manifest)
        dataset = ManifestVideoDataset(records, 'trump', 'test')
        labels = {int(dataset[index][1][0]) for index in range(len(dataset))}
        self.assertEqual(labels, {0, 1})

    def test_missing_selected_media_fails(self):
        self.records[0].path.unlink()
        with self.assertRaisesRegex(ValueError, "missing media file"):
            ManifestVideoDataset(self.records, "trump", "test")

    def test_eval_does_not_require_training_media(self):
        from dataclasses import replace
        missing_train = replace(self.records[0], clip_id="train", path=self.root / "missing.mp4",
                                driver_video_id="different", split="train")
        dataset = ManifestVideoDataset(self.records + [missing_train], "trump", "test")
        self.assertEqual(len(dataset), 2)


if __name__ == "__main__":
    unittest.main()
