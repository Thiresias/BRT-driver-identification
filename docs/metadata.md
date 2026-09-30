# Video metadata

The BRT entry point accepts CSV files or JSON arrays through `--metadata`.
It no longer infers labels or original recordings from directory/file names.
No media files need to be moved. You can keep existing folders or adopt
`genuine/speaker_id/video_id/clip_id.mp4`.

## Schema

Each row describes one **already extracted, cropped video file**. Fields marked
required must be non-empty. IDs are case-sensitive strings; quote numeric IDs
in JSON so leading zeros are preserved. Missing timestamps can be blank/null or
a pair of NaN values; internally they become `None`. A single missing endpoint
is rejected. Use consistent identities across all
manifests (for example, `trump`, not sometimes `Trump`).

| Field | Required | Meaning |
| --- | --- | --- |
| `clip_id` | Yes | Globally unique asset ID across the supplied manifests. |
| `path` | Yes | Absolute media path, or path relative to this manifest's directory. |
| `kind` | Yes | `genuine` or `generated`. |
| `driver_id` | For named drivers | Actual person providing the facial behavior. Blank for unresolved `non_poi` drivers. |
| `driver_scope` | No | `named` (default), or `non_poi` for an explicitly known negative population. |
| `poi_id` | For `non_poi` | POI for whom these drivers are known negatives. Prevents reusing `world` with another POI accidentally. |
| `appearance_id` | No | Visible face identity when known; equals `driver_id` for a named genuine video. |
| `driver_video_id` | No | Original recording supplying the motion, shared by its clips and generated derivatives. Leave blank when unknown; overlap checks are then incomplete. |
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

For named drivers, labels are computed as `int(driver_id == poi)`. For `--poi trump`, a Trump-driven
Obama face is positive, while an Obama-driven Trump face is negative. `CDF` is
a dataset, not an identity: use an actual speaker ID such as `cdf:id0`.
Do not guess unknown driver identities from appearance. For known negative
populations, set `driver_scope=non_poi`, leave `driver_id` empty, and set
`poi_id` to the selected POI. These records receive label 0 only for that POI.
`world` must exclude the POI: the script cannot verify this assertion from pixels.
No filename substring, including `drivingCDF`, controls the label.

## Grouping and split validation

- Genuine clips with the same known driver/recording key are grouped into
  one recording. For `non_poi` records, the key uses `poi_id` and the explicit
  recording ID rather than pretending all unknown speakers are one identity. Each of ten windows chooses one clip from that recording.
- Each generated file is its own evaluation item, even if it shares its driver
  recording with another generated file. Its ten windows share its group ID.
- The same `(driver_id, driver_video_id)` cannot occur in different splits,
  including genuine/generated derivatives when provenance is known. Missing
  recording IDs emit a warning; these assets remain separate and cannot be
  checked for recording overlap. Unresolved `world` identities also cannot
  automatically be matched to named speakers in other manifests. Split original recordings first,
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

## Generate a CSV from your folders

From the repository root, for a dataset containing `data/trump/` and optionally
`data/world/`, run:

```bash
python -m tools.create_metadata --root data --poi trump
```

The CSV generator and shared schema/validation code live in `tools/`:
`create_metadata.py` and `metadata.py`. The video decoder remains in
`LIA_encoder/metadata_dataset.py` and imports this shared validation module.

You can also run both scripts directly from inside `tools/`:

```bash
cd tools
python create_metadata.py --root ../data --poi trump
python metadata.py --metadata ../data/trump/metadata.csv --poi trump
```

Command-line paths are relative to your current working directory. Video paths
stored in the CSV are relative to the CSV's directory. From the repository root,
use `python -m tools.create_metadata` / `python -m tools.metadata`; from inside
`tools/`, use the script commands above (or `python -m create_metadata` /
`python -m metadata`). The former `LIA_encoder.metadata` module has moved.

The accepted pre-split layout is:

| Location | Interpreted content | Label for Trump |
| --- | --- | --- |
| `data/trump/train/` | Genuine Trump clips | 1 |
| `data/trump/val/` | Genuine Trump validation clips | 1 |
| `data/trump/test/` | Generated videos driven by Trump | 1 |
| `data/world/train/` | Genuine videos of non-Trump people | 0 |
| `data/world/val/` | Genuine non-Trump validation videos | 0 |
| `data/world/test/` | Generated videos driven by non-Trump people | 0 |

`val` is written as `validation`; a folder already named `validation` is also
accepted. Subfolders are scanned recursively. Supported extensions are MP4,
AVI, MOV, MKV, WEBM, and M4V (case-insensitive). Other files are ignored. The
script uses folder membership as your assertion of kind and driver membership;
it does not inspect identities or distinguish genuine/fake content from pixels.
Do not place genuine evaluation videos in `test` with this generator convention;
use a manually curated manifest for a different protocol.

For arbitrary input locations, or a POI-only/world-only manifest:

```bash
python -m tools.create_metadata --poi-dir /datasets/trump --world-dir /datasets/others --poi trump --output metadata.csv
python -m tools.create_metadata --world-dir /datasets/others --poi trump --output world.csv
```

A single-population CSV is allowed when generating. Supply the positive and
negative manifests together for binary training/evaluation. The loader still
requires both classes in each selected split.

For genuine videos, `151_part_1 [11.40 - 29.76].mp4` produces recording ID
`trump/151` and timestamps `11.40,29.76`. `151_part_2.mp4` shares that recording
ID and has empty timestamps (pandas reads these as NaN). With no `_part_N`
suffix, the filename stem becomes the recording ID. Relative subfolders are
included in recording IDs, keeping `person_a/001.mp4` separate from
`person_b/001.mp4` in `world`. Train/val folder names are excluded from recording
IDs, so accidental cross-split recordings can be detected. Verify this naming
convention fits your data; the generator cannot detect unrelated recordings
with indistinguishable filenames. Clip IDs contain the POI name (or `world`) followed by a colon and five lowercase
hexadecimal characters, e.g. `trump:1d54a` or `world:a80f2`. The suffix ranges from
`00000` to `fffff`,
with capacity for 1,048,576 clips per CSV. They derive from role and relative
path; collisions are resolved deterministically within the generated CSV.
Regenerating the same collection gives the same IDs. Adding/removing clips can
change IDs involved in collisions. Generate POI and world together when possible:
separately generated CSVs may have colliding IDs, which the loader rejects.

Generated assets retain a blank `driver_video_id` and `appearance_id` rather
than inferring them from the generated filename. Recognized timestamp suffixes
are preserved, but must refer to the original driver interval; review them.
Generation prints a warning that recording-overlap validation is incomplete.
You can fill in original recording IDs and visible identities afterward to
improve provenance checks.

If genuine clips are **not already split**, explicitly request a validation
fraction:

```bash
python -m tools.create_metadata --root data --poi trump --val-fraction 0.2 --seed 42
```

Place unsplit genuine clips directly inside each population folder (nested
subfolders are allowed). An optional `test/` subfolder still contains explicitly
supplied generated videos. At least two original genuine recordings per supplied
population are required. The script assigns whole recordings deterministically
to train/validation, separately for POI and world; no video files move. The
fraction is approximate for small datasets, and at least one recording remains
in each split. Do not combine `--val-fraction` with existing train/val folders.
Without this option, videos outside named splits are rejected, not silently
assigned. For differing layouts, generate separate manifests in separate calls.

By default, the CSV is written inside the POI folder as `metadata.csv`, including
when world videos are supplied: `--root data --poi trump` writes
`data/trump/metadata.csv`. World-only generation writes `metadata.csv` inside
the world input folder. `--output` remains available to choose another path.
Media paths are computed relative to the actual CSV location, including world
videos outside the POI folder.

Existing CSVs are protected unless you pass `--force`. The output is validated
before replacement. The generator needs only the Python standard library and
does not open media; dataset construction later validates the actual videos.

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
python -m tools.metadata --metadata examples/metadata.csv --poi trump
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
