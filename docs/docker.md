# Docker

This image packages the BRT classifier, metadata tools and tests. It targets
Linux x86_64 with Python 3.10, PyTorch 2.5.1 / torchvision 0.20.1 and CUDA 12.4
wheels. This is a compatibility baseline, not a claim to reproduce the paper's
original environment. Direct dependencies are pinned in `docker/constraints.txt`;
transitive dependencies and the Python base image are not fully locked.
Upstream LIA training/demo/evaluation and face-cropping utilities may need
additional dependencies; the supported workflow here starts with cropped videos.

## Build and check

Install Docker. GPU training/evaluation also needs a compatible NVIDIA GPU and
host driver, plus the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
configured for Docker. CUDA is provided inside the image; a host Conda environment
or CUDA toolkit is unnecessary. This CUDA 12.4 baseline may need updating for
new GPU architectures. The matched PyTorch versions are documented
[upstream](https://pytorch.org/get-started/previous-versions/).

From the repository root:

```bash
docker build -t brt-driver-identification:local .
docker run --rm brt-driver-identification:local
docker run --rm --gpus all brt-driver-identification:local \
  python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

The build runs dependency checks, the classifier's import/CLI check and the
unittest suite without a GPU or model weights. It downloads several GB of
packages. The default command shows classifier help. A successful build does
not validate GPU training or checkpoint compatibility; run the GPU check and
then a short training run with your data.

## Data and checkpoints

Mount datasets and model weights at runtime; they are excluded from the image.
The examples assume `data/`, `LIA_encoder/checkpoints/vox.pt`, and `output/` in
your current repository directory. Obtain `vox.pt` using the link in
[the LIA README](../LIA_encoder/README.md#1-animation-demo).

```bash
mkdir -p output
test -f LIA_encoder/checkpoints/vox.pt
```

Relative video paths in CSVs work when the entire dataset tree is mounted at
`/data`. Absolute paths must exist **inside the container**: regenerate the
manifest inside Docker, or mount the dataset at its original absolute path.
Mount all directories referenced by your manifests, including `world`.

Generate and validate metadata without a GPU (the data mount is writable because
these commands create CSV/report files):

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$(pwd)/data",dst=/data \
  brt-driver-identification:local \
  python -m tools.create_metadata --root /data --poi trump

docker run --rm --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$(pwd)/data",dst=/data \
  brt-driver-identification:local \
  python -m tools.validate_videos --metadata /data/trump/metadata.csv \
  --poi trump --output /data/trump/metadata.validated.csv
```

## Train

```bash
docker run --rm --gpus '"device=0"' --shm-size=16g \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$(pwd)/data",dst=/data,readonly \
  --mount type=bind,src="$(pwd)/LIA_encoder/checkpoints",dst=/app/LIA_encoder/checkpoints,readonly \
  --mount type=bind,src="$(pwd)/output",dst=/app/output \
  brt-driver-identification:local \
  python head_mvt_classification_LIA.py \
  --metadata /data/trump/metadata.validated.csv --poi trump \
  --name trump --training --epochs 101
```

Only host GPU 0 is exposed. The loader currently uses 16 workers and batches of
four groups of ten windows: its shared-memory demand is substantial. The example
raises Docker's shared-memory limit to 16 GiB; the host still needs sufficient
RAM. Increase it if workers report shared-memory bus errors. Results persist in
`output/` on the host. `--user` gives output files your host UID/GID (Linux).
Use the same `--sampling` and `--max-stride` for validation and training.

## Evaluate

Use the same mounts and GPU settings, omit `--training`, give the run a new name,
and supply the trained classifier checkpoint:

```bash
docker run --rm --gpus '"device=0"' --shm-size=16g \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$(pwd)/data",dst=/data,readonly \
  --mount type=bind,src="$(pwd)/LIA_encoder/checkpoints",dst=/app/LIA_encoder/checkpoints,readonly \
  --mount type=bind,src="$(pwd)/output",dst=/app/output \
  brt-driver-identification:local \
  python head_mvt_classification_LIA.py \
  --metadata /data/trump/metadata.validated.csv --poi trump \
  --name trump-test --ckpt /app/output/trump/ckpt/latest.pth
```

The default evaluation split is `test`. Both the LIA backbone `vox.pt` and the
trained classifier checkpoint are needed. Metadata preparation and unit tests
can run on CPU; classifier training/evaluation currently require CUDA.

To rerun tests in the image:

```bash
docker run --rm brt-driver-identification:local python -m unittest discover -s tests -v
```
