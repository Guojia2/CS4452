
from __future__ import annotations

import os                          # ← this was missing
import json
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.io import read_video
from PIL import Image


# ---------------------------------------------------------------------------
# Transform factory
# ---------------------------------------------------------------------------

def build_transforms(img_size: int = 224) -> T.Compose:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return T.Compose([
        T.Resize(int(img_size * 256 / 224)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

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
) -> List[Image.Image]:
    frames, _, _ = read_video(
        video_path,
        start_pts=start_sec,
        end_pts=end_sec,
        pts_unit="sec",
    )  # (T, H, W, C)

    if frames.shape[0] == 0:
        raise ValueError(
            f"No frames decoded from {video_path} between {start_sec:.2f}s and {end_sec:.2f}s"
        )

    if frames.shape[0] == 1:
        indices = torch.zeros(num_frames, dtype=torch.long)
    else:
        indices = torch.linspace(0, frames.shape[0] - 1, steps=num_frames).long()

    sampled = frames[indices]  # (num_frames, H, W, C)

    return [Image.fromarray(frame.numpy()) for frame in sampled]


class THUMOSVideoDataset(Dataset):
    def __init__(
        self,
        video_dir: str,
        ann_path: str,
        subset: str = "training",
        clip_len_sec: float = 2.0,
        stride_sec: float = 1.0,
        num_frames: int = 16,
        transform: Optional[Callable] = None,
        img_size: int = 224,
    ) -> None:
        self.video_dir = video_dir
        self.ann_path = ann_path
        self.subset = subset
        self.clip_len_sec = clip_len_sec
        self.stride_sec = stride_sec
        self.num_frames = num_frames
        self.transform = transform or build_transforms(img_size=img_size)

        self.annotations = load_thumos_annotations(ann_path, subset=subset)
        self.clips: List[Tuple[str, float, float]] = self._build_clip_index()

    def _build_clip_index(self) -> List[Tuple[str, float, float]]:
        clips: List[Tuple[str, float, float]] = []

        for video_name, meta in self.annotations.items():
            duration = meta["duration"]
            start = 0.0

            while start + self.clip_len_sec <= duration:
                end = start + self.clip_len_sec
                clips.append((video_name, start, end))
                start += self.stride_sec

        return clips
    
    def __len__(self) -> int:
        return len(self.clips)
    
    def __getitem__(self, idx: int):
        video_name, start, end = self.clips[idx]
        video_path = os.path.join(self.video_dir, f"{video_name}.mp4")

        frames = decode_video_frames(video_path, start, end, self.num_frames)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        clip = torch.stack(frames).permute(1, 0, 2, 3)  # (C, T, H, W)

        return clip, video_name, start, end


# ---------------------------------------------------------------------------
# Secondary dataset — pre-extracted features from local disk
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

        return window, labels 