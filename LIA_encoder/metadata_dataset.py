"""Video sampling for the metadata-driven BRT classifier."""

import torch
from torch.utils.data import Dataset

from tools.metadata import group_records
from tools.video_sampling import (
    NUM_WINDOWS, SamplingConfig, decode_window, probe_video, select_record,
)


class ManifestVideoDataset(Dataset):
    """Ten 40-frame windows per recording/generated video, at any positive FPS.

    Multiscale (default) uses strides 1..max_stride when they fit the clip.
    Fixed mode preserves the eight-second, 5 Hz sampling protocol (FPS >= 5).
    Returns (frames[K,T,C,H,W], labels[K], group_ids[K], fps[K]).
    Videos must be pre-cropped to 256x256; manifest timestamps are provenance.
    """

    K = NUM_WINDOWS

    def __init__(self, records, poi, split, seq_length=8, sampling='multiscale', max_stride=5):
        if seq_length != 8:
            raise ValueError('seq_length is retained for compatibility and must be 8; '
                             'use sampling to choose the 40-frame protocol')
        self.config = SamplingConfig(sampling, max_stride)
        self.groups = group_records(records, split, poi)
        self.video_ids = [group.video_id for group in self.groups]
        self.media_info = {
            record.path: probe_video(record, self.config)
            for group in self.groups for record in group.records
        }

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]
        frames, fps_values = [], []
        for window in range(self.K):
            record, rng = select_record(group.records, window)
            sequence, fps = self.load_vid(record, rng)
            frames.append(sequence)
            fps_values.append(fps)
        return (torch.stack(frames), torch.tensor([group.label] * self.K, dtype=torch.long),
                [group.video_id] * self.K, torch.tensor(fps_values))

    def load_vid(self, record, rng):
        info = self.media_info[record.path]
        frames = decode_window(record, info, rng, self.config)
        return torch.from_numpy(frames).permute(0, 3, 1, 2), info.fps
