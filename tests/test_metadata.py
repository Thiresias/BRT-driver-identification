import csv
import json
import tempfile
import unittest
from pathlib import Path

from LIA_encoder.metadata import REQUIRED_FIELDS, group_records, load_metadata


def row(clip_id="trump-151-a", **changes):
    result = dict(clip_id=clip_id, path=f"media/{clip_id}.mp4", kind="genuine",
                  driver_id="trump", appearance_id="trump", driver_video_id="151",
                  split="train", start_sec=11.4, end_sec=29.76)
    result.update(changes)
    return result


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def manifest(self, rows, name="metadata.json"):
        path = self.root / name
        path.write_text(json.dumps(rows))
        return path

    def negative(self, **changes):
        return row("obama-151-a", driver_id="obama", appearance_id="obama", **changes)

    def test_genuine_grouping_and_identical_basenames(self):
        records = load_metadata(self.manifest([
            row(path="trump/000.mp4"), row("trump-151-b", path="trump/part2.mp4"),
            self.negative(path="obama/000.mp4"),
        ]))
        groups = group_records(records, "train", "trump")
        self.assertEqual(len(groups), 2)
        self.assertEqual(sorted(len(g.records) for g in groups), [1, 2])
        self.assertEqual([g.label for g in groups], [0, 1])
        self.assertEqual(len({g.video_id for g in groups}), 2)

    def test_filename_prefix_does_not_determine_recording(self):
        records = load_metadata(self.manifest([
            row(path="151_part_1.mp4"), row("b", path="151_part_2.mp4", driver_video_id="other"),
            self.negative(),
        ]))
        self.assertEqual(len(group_records(records, "train", "trump")), 3)

    def test_generated_videos_are_separate_and_driver_defines_label(self):
        records = load_metadata(self.manifest([
            row("fake-a", kind="generated", appearance_id="obama", path="drivingCDF/a.mp4"),
            row("fake-b", kind="generated", appearance_id="belkacem"),
            row("fake-c", kind="generated", driver_id="obama", appearance_id="trump"),
        ]))
        groups = group_records(records, "train", "trump")
        self.assertEqual([g.label for g in groups], [1, 1, 0])
        self.assertEqual(len(groups), 3)
        self.assertTrue(all(len(g.records) == 1 for g in groups))

    def test_csv_leading_zeros_and_relative_paths(self):
        path = self.root / "metadata.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REQUIRED_FIELDS)
            writer.writeheader()
            record = row(driver_id="001", appearance_id="001", driver_video_id="0002")
            writer.writerow({k: record[k] for k in REQUIRED_FIELDS})
        record = load_metadata(path)[0]
        self.assertEqual(record.driver_id, "001")
        self.assertEqual(record.driver_video_id, "0002")
        self.assertEqual(record.path, self.root / "media/trump-151-a.mp4")

    def test_paths_relative_to_each_manifest(self):
        (self.root / "sub").mkdir()
        a = self.manifest([row()], "a.json")
        b = self.manifest([self.negative()], "sub/b.json")
        records = load_metadata([a, b])
        self.assertEqual(records[1].path.parent, self.root / "sub/media")

    def test_duplicate_id_and_path_rejected(self):
        for second, message in [(row(path="another.mp4"), "duplicate clip_id"),
                                (row("another", path="media/trump-151-a.mp4"), "duplicate media path")]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                load_metadata(self.manifest([row(), second]))

    def test_recording_split_leakage_including_generated_derivative(self):
        for kind in ["genuine", "generated"]:
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "appears in both"):
                load_metadata([self.manifest([row()], "train.json"), self.manifest([
                    row("heldout", kind=kind, split="test")], "test.json")])

    def test_missing_and_invalid_fields_rejected(self):
        for changes in [dict(driver_id=""), dict(kind="fake"), dict(split="dev"),
                        dict(driver_video_id=151), dict(appearance_id="obama"),
                        dict(start_sec=-1), dict(end_sec=0), dict(end_sec="nan"),
                        dict(start_sec="", end_sec=10)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                load_metadata(self.manifest([row(**changes)]))

    def test_empty_missing_split_and_one_class_rejected(self):
        with self.assertRaisesRegex(ValueError, "no records"):
            load_metadata(self.manifest([]))
        records = load_metadata(self.manifest([row()]))
        with self.assertRaisesRegex(ValueError, "no records"):
            group_records(records, "test", "trump")
        with self.assertRaisesRegex(ValueError, "both POI"):
            group_records(records, "train", "trump")

    def test_timestamps_preserved_as_provenance(self):
        record = load_metadata(self.manifest([row(start_sec=211.04, end_sec=225.76)]))[0]
        self.assertEqual((record.start_sec, record.end_sec), (211.04, 225.76))

    def test_case_sensitive_poi_and_reusable_labels(self):
        records = load_metadata(self.manifest([row(), self.negative()]))
        self.assertEqual([g.label for g in group_records(records, "train", "obama")], [1, 0])
        with self.assertRaisesRegex(ValueError, "case-sensitive"):
            group_records(records, "train", "Trump")


if __name__ == "__main__":
    unittest.main()
