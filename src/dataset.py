import os
import json
import torch
import numpy as np
import av  # PyAV for video decoding
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Optional


# ---------------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------------

def load_thumos_annotations(ann_path: str, subset: str = None) -> Dict:
    """
    Loads thumos_14_anno.json from OpenTAD.
    Optionally filter by subset: "training", "validation", or "test".
    Note: THUMOS14 calls the training split "training" in this file,
    which corresponds to the validation videos (val folder) on disk.
    """
    with open(ann_path) as f:
        data = json.load(f)

    database = data["database"]

    if subset is not None:
        database = {
            k: v for k, v in database.items()
            if v.get("subset") == subset
        }

    return database


# ---------------------------------------------------------------------------
# Label map — THUMOS-14 classes
# ---------------------------------------------------------------------------

THUMOS14_CLASSES = [
    "BaseballPitch", "BasketballDunk", "Billiards", "CleanAndJerk",
    "CliffDiving", "CricketBowling", "CricketShot", "Diving",
    "FrisbeeCatch", "GolfSwing", "HammerThrow", "HighJump",
    "JavelinThrow", "LongJump", "PoleVault", "Shotput",
    "SoccerPenalty", "TennisSwing", "ThrowDiscus", "VolleyballSpiking",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(THUMOS14_CLASSES)}


# ---------------------------------------------------------------------------
# Raw Video Dataset  (used during feature extraction)
# ---------------------------------------------------------------------------

def decode_video_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    num_frames: int,
) -> torch.Tensor:
    """
    Decode `num_frames` evenly-spaced frames from [start_sec, end_sec].
    Returns a tensor of shape (C, T, H, W) in float32, normalized to [0, 1].
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    duration = float(stream.duration * stream.time_base)

    start_sec = max(0.0, start_sec)
    end_sec   = min(duration, end_sec)

    timestamps = np.linspace(start_sec, end_sec, num_frames)
    target_pts = [int(t / stream.time_base) for t in timestamps]

    frames = []
    for pts in target_pts:
        container.seek(pts, stream=stream)
        for frame in container.decode(stream):
            img = frame.to_ndarray(format="rgb24")  # (H, W, C)
            frames.append(img)
            break

    container.close()

    # Stack → (T, H, W, C) → (C, T, H, W)
    frames = np.stack(frames, axis=0).astype(np.float32) / 255.0
    frames = torch.from_numpy(frames).permute(3, 0, 1, 2)
    return frames


class THUMOSVideoDataset(Dataset):
    """
    Iterates over sliding-window clips of each video for feature extraction.
    Each item is (clip_tensor, video_name, window_start_sec, window_end_sec).
    """

    def __init__(
        self,
        video_dir: str,
        ann_path: str,
        clip_len_sec: float = 2.0,
        stride_sec: float = 1.0,
        num_frames: int = 16,
        subset: str = "training",   # add this

        transform=None,
    ):
        self.video_dir    = video_dir
        self.clip_len_sec = clip_len_sec
        self.stride_sec   = stride_sec
        self.num_frames   = num_frames
        self.transform    = transform

        annotations = load_thumos_annotations(ann_path, subset = subset)
        self.clips: List[Tuple] = []

        for video_name, meta in annotations.items():
            video_path = os.path.join(video_dir, f"{video_name}.mp4")
            if not os.path.exists(video_path):
                # Try .avi as THUMOS has mixed formats
                video_path = video_path.replace(".mp4", ".avi")
            if not os.path.exists(video_path):
                continue

            duration = meta["duration"]
            start = 0.0
            while start + clip_len_sec <= duration:
                self.clips.append((video_path, video_name, start, start + clip_len_sec))
                start += stride_sec

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        video_path, video_name, start, end = self.clips[idx]
        clip = decode_video_frames(video_path, start, end, self.num_frames)
        if self.transform:
            clip = self.transform(clip)
        return clip, video_name, start, end


# ---------------------------------------------------------------------------
# Feature Dataset  (used for temporal head training)
# ---------------------------------------------------------------------------

class THUMOSFeatureDataset(Dataset):
    """
    Loads pre-extracted clip-level or frame-level features from disk.

    Each .pt file per video has shape (N_clips, feature_dim).
    The dataset returns fixed-length windows of consecutive clip features
    along with their ground-truth labels for the temporal detection task.
    """

    def __init__(
        self,
        feature_dir: str,
        ann_path: str,
        window_size: int = 128,   # number of clips per training sample
        stride: int = 64,
        subset: str = "training",   

        split: str = "val",       # THUMOS uses "val" for training TAD models
    ):
        self.feature_dir = feature_dir
        self.window_size = window_size
        self.stride      = stride

        annotations = load_thumos_annotations(ann_path, subset = subset)
        self.samples: List[Dict] = []

        for video_name, meta in annotations.items():
            feat_path = os.path.join(feature_dir, f"{video_name}.pt")
            if not os.path.exists(feat_path):
                continue

            features = torch.load(feat_path)          # (N_clips, D)
            n_clips  = features.shape[0]
            gt_segs  = meta["annotations"]            # list of {label, segment}

            start = 0
            while start + window_size <= n_clips:
                self.samples.append({
                    "feat_path": feat_path,
                    "start":     start,
                    "gt_segs":   gt_segs,
                    "duration":  meta["duration"],
                    "n_clips":   n_clips,
                })
                start += stride

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        features = torch.load(s["feat_path"])                    # (N_clips, D)
        window   = features[s["start"]: s["start"] + self.window_size]  # (W, D)

        # Build a per-clip label vector  (multi-label, one-hot)
        num_classes = len(THUMOS14_CLASSES)
        labels = torch.zeros(self.window_size, num_classes)

        clips_per_sec = s["n_clips"] / s["duration"]
        for ann in s["gt_segs"]:
            cls_idx = CLASS_TO_IDX.get(ann["label"], -1)
            if cls_idx == -1:
                continue
            seg_start, seg_end = ann["segment"]
            # Convert seconds → clip indices
            c_start = int(seg_start * clips_per_sec) - s["start"]
            c_end   = int(seg_end   * clips_per_sec) - s["start"]
            c_start = max(0, c_start)
            c_end   = min(self.window_size, c_end)
            if c_start < c_end:
                labels[c_start:c_end, cls_idx] = 1.0

        return window, labels   # (W, D), (W, num_classes)