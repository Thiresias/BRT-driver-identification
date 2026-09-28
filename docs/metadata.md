# Video metadata

The BRT entry point accepts CSV files or JSON arrays through `--metadata`.
It no longer infers labels or original recordings from directory/file names.
No media files need to be moved. You can keep existing folders or adopt
`genuine/speaker_id/video_id/clip_id.mp4`.

## Schema

Each row describes one **already extracted, cropped video file**. Fields marked
required must be non-empty. IDs are case-sensitive strings; quote numeric IDs
in JSON so leading zeros are preserved. Use consistent identities across all
manifests (for example, `trump`, not sometimes `Trump`).

| Field | Required | Meaning |
| --- | --- | --- |
| `clip_id` | Yes | Globally unique asset ID across the supplied manifests. |
| `path` | Yes | Absolute media path, or path relative to this manifest's directory. |
| `kind` | Yes | `genuine` or `generated`. |
| `driver_id` | Yes | Actual person providing the facial behavior. |
| `appearance_id` | Yes | Visible face identity; equals `driver_id` for genuine video. |
| `driver_video_id` | Yes | Original recording supplying the motion, shared by its clips and generated derivatives. |
| `split` | Yes | `train`, `validation`, or `test`. |
| `start_sec`, `end_sec` | No | Both present or both absent; interval in the original driver recording. |
| `source_asset_id` | No | Appearance image/video used to generate the fake. |
| `method` | No | Deepfake generation method/version. |

Timestamps are **provenance only**. The loader samples within the file at `path`;
it does not seek to these original-recording timestamps or extract clips from
full recordings. For `151_part_1 [57.38 - 75.70].mp4`, use recording ID `151`
and interval `57.38,75.70`. Include the interval in `clip_id` because part numbers
can repeat. Namespace recording IDs by dataset when the same speaker has
independent datasets that reuse numbers.

Labels are computed as `int(driver_id == poi)`. For `--poi trump`, a Trump-driven
Obama face is positive, while an Obama-driven Trump face is negative. `CDF` is
a dataset, not an identity: use an actual speaker ID such as `cdf:id0`.
Do not guess unknown driver identities from appearance. This version requires
resolved driver identities before an asset can participate in the experiment.
No filename substring, including `drivingCDF`, controls the label.

## Grouping and split validation

- Genuine clips with the same `(driver_id, driver_video_id)` are grouped into
  one recording. Each of ten windows chooses one clip from that recording.
- Each generated file is its own evaluation item, even if it shares its driver
  recording with another generated file. Its ten windows share its group ID.
- The same `(driver_id, driver_video_id)` cannot occur in different splits,
  including genuine/generated derivatives. Split original recordings first,
  then assign all derived assets to that split.
- Duplicate `clip_id` values and duplicate resolved media paths are errors.
- Each selected split must contain both POI and non-POI drivers for the current
  binary training/evaluation workflow. Missing splits and misspelled POI IDs
  fail early instead of producing undefined AUC.
- These checks operate only on the supplied metadata. Supply all experiment
  manifests together to check cross-split overlap. They cannot detect copied
  media with deliberately different identities, paths, or recording IDs.

Recordings and clips are sorted by metadata IDs, making grouping independent
of row order. Sampling retains ten deterministic windows, using local random
generators rather than resetting NumPy's global random state. This patch does
not introduce epoch-varying augmentation.

## Example and migration

[`../examples/metadata.csv`](../examples/metadata.csv) contains illustrative
paths and identities, not downloadable research data. Replace them with your
actual assets and complete the original-recording provenance of generated
videos. Keep `drivingTrump_sourceObama` folders if useful, but explicitly write
`driver_id=trump,appearance_id=obama` in the manifest.

An equivalent JSON record is:

```json
[
  {
    "clip_id": "trump-151-part1-57.38-75.70",
    "path": "../data/trump/151_part_1 [57.38 - 75.70].mp4",
    "kind": "genuine",
    "driver_id": "trump",
    "appearance_id": "trump",
    "driver_video_id": "trump:151",
    "split": "train",
    "start_sec": 57.38,
    "end_sec": 75.70
  }
]
```

That single record illustrates syntax; add negative drivers and held-out
recordings before running the classifier. Filenames alone generally cannot
recover the exact driver recording and generation provenance of a deepfake,
so this patch does not automatically invent those mappings.

Validate structure, labels, and splits without PyTorch or access to the videos:

```bash
python -m LIA_encoder.metadata --metadata examples/metadata.csv --poi trump
```

When the dataset is instantiated, it additionally checks selected media exists,
opens correctly, is pre-cropped to 256x256, has approximately 25 or 30 FPS, and
contains at least eight seconds. Invalid clips produce an explicit error instead
of being silently dropped. Failed/short frame decoding is also reported.
Timestamps do not substitute for these checks on actual media.

Training reads `train` and, by default, `validation`; standalone evaluation reads
only `test`. `--eval_split` can select `validation` or `test` explicitly. No dummy
training directories or files are needed for evaluation. Multiple files are
supported, for example `--metadata genuine.csv generated.json`.

Training saves normalized metadata (including resolved media paths) in
`output/<name>/metadata.json` and the selected POI/split in `data_config.json`.
If sharing that snapshot, replace machine-specific absolute paths as needed.

## Regression tests

```bash
python -m unittest discover -s tests -v
```

The metadata tests require only Python's standard library. Synthetic-video
integration tests also require `torch`, `numpy`, and `opencv-python-headless`
(or an existing compatible OpenCV install). Those tests run on CPU and need no
research data or model checkpoints; they are skipped if dependencies are absent.

This patch changes metadata, grouping, labels, and data selection. The remaining
evaluation issues identified in the audit (training-mode validation and the
different AUC scoring rules) still need a separate correction before treating
training-time metrics as a validated reproduction of the paper.
