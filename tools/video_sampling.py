"""Shared window planning and decoding for preflight validation and training."""

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

NUM_FRAMES = 40
NUM_WINDOWS = 10


class VideoError(ValueError):
    pass


@dataclass(frozen=True)
class SamplingConfig:
    mode: str = 'multiscale'
    max_stride: int = 5

    def __post_init__(self):
        if self.mode not in {'multiscale', 'fixed'}:
            raise ValueError('sampling must be multiscale or fixed')
        if not isinstance(self.max_stride, int) or self.max_stride < 1:
            raise ValueError('max_stride must be a positive integer')


@dataclass(frozen=True)
class VideoInfo:
    count: int
    fps: float


@dataclass(frozen=True)
class Window:
    start: int
    offsets: tuple

    @property
    def span(self):
        return self.offsets[-1] + 1


def fail(record, message):
    raise VideoError(f'{record.clip_id} ({record.path}): {message}')


def required_span(fps, config):
    if not math.isfinite(fps) or fps <= 0:
        raise VideoError(f'invalid FPS {fps}; expected a finite positive value')
    if config.mode == 'multiscale':
        return NUM_FRAMES
    if fps < 5:
        raise VideoError(f'fixed sampling needs FPS >= 5 for 40 distinct frames; got {fps}')
    return math.ceil(8 * fps)


def check_info(info, config):
    required = required_span(info.fps, config)
    if not math.isfinite(info.count) or info.count < required:
        raise VideoError(f'frames={info.count}, fps={info.fps}, sampling={config.mode}: '
                         f'need at least {required} source frames')
    return required


def plan_window(info, rng, config):
    """Inclusive last valid start; exactly 40 source frames are valid at stride 1."""
    minimum = check_info(info, config)
    if config.mode == 'multiscale':
        max_stride = min(config.max_stride, (info.count - 1) // (NUM_FRAMES - 1))
        stride = int(rng.randint(1, max_stride + 1))
        offsets = tuple(index * stride for index in range(NUM_FRAMES))
        span = offsets[-1] + 1
    else:
        offsets = tuple(math.floor(index * info.fps / 5) for index in range(NUM_FRAMES))
        span = minimum
    start = int(rng.randint(info.count - span + 1))
    return Window(start, offsets)


def select_record(records, window_index):
    # Both validator and dataset consume RNG draws in exactly the same order.
    rng = np.random.RandomState(window_index)
    return records[int(rng.randint(len(records)))], rng


def probe_video(record, config):
    if not Path(record.path).is_file():
        fail(record, 'missing media file')
    cap = cv2.VideoCapture(str(record.path))
    try:
        if not cap.isOpened():
            raise VideoError('cannot open video')
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        check_info(VideoInfo(count, fps), config)
        info = VideoInfo(int(count), fps)
        check_info(info, config)
        size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if size != (256, 256):
            raise VideoError(f'expected pre-cropped 256x256 video, got {size}')
        return info
    except (VideoError, cv2.error) as exc:
        fail(record, str(exc))
    finally:
        cap.release()


def decode_window(record, info, rng, config):
    # Validate before allocating a capture or referring to a starting frame.
    try:
        window = plan_window(info, rng, config)
    except VideoError as exc:
        fail(record, str(exc))
    cap = cv2.VideoCapture(str(record.path))
    frames = []
    try:
        if not cap.isOpened() or not cap.set(cv2.CAP_PROP_POS_FRAMES, window.start):
            fail(record, f'cannot seek to frame {window.start}')
        selected = set(window.offsets)
        for offset in range(window.span):
            ok, frame = cap.read()
            if not ok or frame is None:
                fail(record, f'decoding failed at frame {window.start + offset}; '
                     f'frames={info.count}, fps={info.fps}, start={window.start}, span={window.span}')
            if frame.shape != (256, 256, 3):
                fail(record, f'unexpected frame shape {frame.shape} at {window.start + offset}')
            if offset in selected:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except cv2.error as exc:
        fail(record, f'OpenCV decoding error: {exc}')
    finally:
        cap.release()
    if len(frames) != NUM_FRAMES:
        fail(record, f'expected {NUM_FRAMES} frames, decoded {len(frames)}')
    return np.stack(frames)


def full_decode(record, info):
    """Check every advertised frame and compare decoded count with the header."""
    cap = cv2.VideoCapture(str(record.path))
    decoded = 0
    try:
        if not cap.isOpened():
            fail(record, 'cannot open video for full decoding')
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame.shape != (256, 256, 3):
                fail(record, f'unexpected frame shape at frame {decoded}: {frame.shape}')
            decoded += 1
        if decoded != info.count:
            fail(record, f'full decode returned {decoded} frames; header reports {info.count}')
    except cv2.error as exc:
        fail(record, f'OpenCV full-decode error: {exc}')
    finally:
        cap.release()
