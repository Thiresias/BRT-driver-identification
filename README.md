<!-- omit in toc -->
# BRT-driver-identification
Repository of the paper: "Who is driving this deepfake? Beyond Deepfake Detection With Driver Identification"

![alt text](assets/driver_identification.jpg)

<!-- omit in toc -->
## Table of Contents
- [Installation](#installation)
  - [Libraries](#python-environment)
  - [Dataset metadata](#dataset-metadata-and-preprocessing)
- [Scripts](#scripts)
  - [Training](#training)
  - [Testing](#driver-identification-testing)
- [Citation](#citation)
- [Contributing](#contributing)


<!-- omit in toc -->
## Installation

### Python environment
To setup your Python environment, we recommend using Conda:
```bash
# Create your environment (works fine with python 3.10)
conda create -n brt python=3.10

# Install PyTorch first (check your CUDA version on https://pytorch.org/get-started/previous-versions/)
pip install torch torchvision

# Install the remaining dependencies
pip install -r requirements.txt
```

### Dataset metadata and preprocessing

Crop videos using the preprocessing script `crop_video.py` from
[FOMM](https://github.com/AliaksandrSiarohin/first-order-model). The current
classifier expects 256x256 videos. Multiscale sampling accepts any finite positive
FPS and requires at least 40 source frames. It samples 40 frames at a random
interval of 1–5 frames (limited by clip length).

Describe the videos in CSV or JSON metadata instead of passing positive/negative
folders. Each record specifies the driver identity, visible identity, original
driver recording, and split. The label is positive exactly when `driver_id`
matches `--poi`; folder names such as `drivingCDF` no longer determine labels.

Create a manifest automatically from `<root>/<poi>/{train,val,test}` and optional
`<root>/world/{train,val,test}` folders:

```bash
python -m tools.create_metadata --root data --poi trump
```

The CSV is saved to `data/trump/metadata.csv` by default. Clip IDs include the population prefix and five hexadecimal characters, such as
`trump:1d54a` or `world:1d54a`, with collisions resolved within the generated CSV.

Train/val folders contain genuine clips; test folders contain generated videos
with the corresponding drivers. Missing timestamps remain blank (NaN in pandas).
`world` is explicitly scoped as non-POI, not a single speaker identity. Unknown
generated-video provenance is reported as incomplete overlap validation.

See [the metadata schema and migration guide](docs/metadata.md) and
[example CSV](examples/metadata.csv). Existing video paths can be retained.
Group clips from the same original recording into the same split, including
any generated derivatives. Training uses `train` and `validation`; standalone
evaluation uses `test` and requires no training videos.

Check your metadata before loading videos or model weights:

```bash
python -m tools.metadata --metadata data/trump/metadata.csv --poi trump
```

Validate actual video decoding before training:

```bash
python -m tools.validate_videos --metadata data/trump/metadata.csv --poi trump --output data/trump/metadata.validated.csv
```

This creates a filtered CSV, a rejection CSV, and a JSON validation summary.
Train with `--metadata data/trump/metadata.validated.csv`. Use `--full-decode`
for an additional full sequential decode. Use the same `--sampling` and
`--max-stride` in validation and training; defaults are `multiscale` and `5`.
The optional `--sampling fixed` mode uses 40 samples at 5 Hz from an eight-second
window and requires FPS >= 5. Multiscale duration varies with FPS and stride.

The frozen LIA backbone requires `LIA_encoder/checkpoints/vox.pt`; see the
[upstream checkpoint download instructions](LIA_encoder/README.md#1-animation-demo).
This is distinct from the trained BRT classifier checkpoint supplied to `--ckpt`.

<!-- omit in toc -->
## Scripts

Run commands from the repository root. Training currently uses one CUDA GPU.
Use your actual POI identity and manifest paths in the following commands.

### Training

```bash
export CUDA_VISIBLE_DEVICES=0
python head_mvt_classification_LIA.py --metadata data/trump/metadata.csv --poi trump --epochs 101 --name trump --training
```

You can supply several manifests: `--metadata genuine.csv generated.json`.
They are validated together, including checks for original driver recordings
appearing in different splits. Metadata paths are resolved relative to each
manifest, not the current working directory.

### Driver identification (Testing)

```bash
export CUDA_VISIBLE_DEVICES=0
python head_mvt_classification_LIA.py --metadata data/trump/metadata.csv --poi trump --name trump-test --ckpt output/trump/ckpt/latest.pth
```

`--ckpt` is required for testing. `--eval_split validation` selects validation
instead of the default test split. During training the default is validation.

<!-- omit in toc -->
## Citation
```
@inproceedings{libourel2025driving,
  title={Who is driving this deepfake? Beyond Deepfake Detection with Driver Identification},
  author={Libourel, Alexandre and Dugelay, Jean-Luc},
  booktitle={2025 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2025},
  organization={IEEE}
}
```
<!-- omit in toc -->
## Contributing

Part of the code is adapted from [LIA](https://github.com/wyhsirius/LIA). We thank authors for their contribution to the community.
