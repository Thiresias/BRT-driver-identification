"""Video sampling for the metadata-driven BRT classifier."""

import math

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .metadata import group_records


class ManifestVideoDataset(Dataset):
    """Ten eight-second windows per genuine recording or generated video.

    Returns (frames[K,T,C,H,W], labels[K], group_ids[K], fps[K]), matching
    the classifier's existing DataLoader contract. Videos must already be
    cropped to 256x256. Manifest timestamps are provenance, not seek offsets.
    """

    K = 10

    def __init__(self, records, poi, split, seq_length=8):
        if seq_length != 8:
            raise ValueError("the current classifier requires eight-second / 40-frame windows")
        self.seq_length = seq_length
        self.groups = group_records(records, split, poi)
        self.video_ids = [group.video_id for group in self.groups]
        self.media_info = {}
        for group in self.groups:
            for record in group.records:
                if not record.path.is_file():
                    raise ValueError(f"{record.clip_id}: missing media file {record.path}")
                cap = cv2.VideoCapture(str(record.path))
                try:
                    if not cap.isOpened():
                        raise ValueError(f"{record.clip_id}: cannot open {record.path}")
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if not math.isfinite(fps) or min(abs(fps - 25), abs(fps - 30)) >= 0.1:
                        raise ValueError(f"{record.clip_id}: expected 25 or 30 FPS, got {fps}")
                    if not math.isfinite(count) or count < seq_length * round(fps):
                        raise ValueError(f"{record.clip_id}: video must be at least {seq_length} seconds")
                    size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if size != (256, 256):
                        raise ValueError(f"{record.clip_id}: expected pre-cropped 256x256 video, got {size}")
                    self.media_info[record.path] = (int(count), fps)
                finally:
                    cap.release()

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]
        frames, fps_values = [], []
        for window in range(self.K):
            # Preserve repeatable ten-window sampling without changing global RNG state.
            rng = np.random.RandomState(window)
            record = group.records[rng.randint(len(group.records))]
            sequence, fps = self.load_vid(record, rng)
            frames.append(sequence)
            fps_values.append(fps)
        return (
            torch.stack(frames),
            torch.tensor([group.label] * self.K, dtype=torch.long),
            [group.video_id] * self.K,
            torch.tensor(fps_values),
        )

    def load_vid(self, record, rng):
        count, fps = self.media_info[record.path]
        window_frames = self.seq_length * round(fps)
        start = rng.randint(count - window_frames + 1)
        stride = round(fps) // 5
        frames = []
        cap = cv2.VideoCapture(str(record.path))
        try:
            if not cap.isOpened() or not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
                raise ValueError(f"{record.clip_id}: cannot seek to frame {start}")
            for offset in range(window_frames):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"{record.clip_id}: decoding failed at frame {start + offset}")
                if offset % stride == 0:
                    if frame.shape != (256, 256, 3):
                        raise ValueError(f"{record.clip_id}: unexpected frame shape {frame.shape}")
                    frames.append(torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        finally:
            cap.release()
        if len(frames) != 40:
            raise ValueError(f"{record.clip_id}: expected 40 sampled frames, got {len(frames)}")
        return torch.stack(frames).permute(0, 3, 1, 2), fps
