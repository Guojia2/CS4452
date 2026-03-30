
from __future__ import annotations

import os                          # ← this was missing
import json
from typing import Callable, Dict, List, Optional, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import av
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
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    # Seek near the start and decode frames in range
    all_frames = []
    container.seek(int(start_sec * av.time_base), any_frame=False)
    for frame in container.decode(video=0):
        pts_sec = float(frame.pts * stream.time_base)
        if pts_sec < start_sec - 0.1:
            continue
        if pts_sec > end_sec + 0.1:
            break
        all_frames.append(frame.to_ndarray(format="rgb24"))

    container.close()

    if len(all_frames) == 0:
        raise ValueError(
            f"No frames decoded from {video_path} between {start_sec:.2f}s and {end_sec:.2f}s"
        )

    frames = np.stack(all_frames)  # (T, H, W, C)
    n = frames.shape[0]

    if n == 1:
        indices = np.zeros(num_frames, dtype=np.int64)
    else:
        indices = np.linspace(0, n - 1, num=num_frames).astype(np.int64)

    sampled = frames[indices]  # (num_frames, H, W, C)

    return [Image.fromarray(f) for f in sampled]


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
        window_size: int = 128,
        stride: int = 64,
        subset: str = "training",
        split: str = "val",
        aug_config: Optional[Dict] = None,
        training: bool = False,
    ):
        self.feature_dir = feature_dir
        self.window_size = window_size
        self.stride      = stride
        self.aug_config  = aug_config or {}
        self.training    = training

        annotations = load_thumos_annotations(ann_path, subset=subset)
        self.samples: List[Dict] = []

        # Action segment index for cut-and-paste: {class_name: [(feat_path, clip_start, clip_end, n_clips, duration)]}
        self.action_index: Dict[str, List[Tuple]] = {c: [] for c in THUMOS14_CLASSES}

        for video_name, meta in annotations.items():
            feat_path = os.path.join(feature_dir, f"{video_name}.pt")
            if not os.path.exists(feat_path):
                continue

            features = torch.load(feat_path, weights_only=True)  # (N_clips, D)
            n_clips  = features.shape[0]
            gt_segs  = meta["annotations"]
            duration = meta["duration"]
            clips_per_sec = n_clips / duration

            start = 0
            while start + window_size <= n_clips:
                self.samples.append({
                    "feat_path": feat_path,
                    "start":     start,
                    "gt_segs":   gt_segs,
                    "duration":  duration,
                    "n_clips":   n_clips,
                })
                start += stride

            # Build action segment index for cut-and-paste
            for ann in gt_segs:
                label = ann["label"]
                if label not in CLASS_TO_IDX:
                    continue
                seg_start, seg_end = ann["segment"]
                c_start = int(seg_start * clips_per_sec)
                c_end   = int(seg_end * clips_per_sec)
                if c_end > c_start:
                    self.action_index[label].append(
                        (feat_path, c_start, c_end, n_clips, duration)
                    )

    def __len__(self):
        return len(self.samples)

    def _jitter_start(self, start: int, n_clips: int) -> int:
        """Apply random offset to window start position during training."""
        jitter_range = self.stride // 2
        offset = torch.randint(-jitter_range, jitter_range + 1, (1,)).item()
        new_start = start + offset
        new_start = max(0, min(new_start, n_clips - self.window_size))
        return new_start

    def _cutpaste(
        self, features: torch.Tensor, labels: torch.Tensor, s: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Paste a random rare-class action segment into a background region.

        Picks a random class (weighted toward underrepresented ones),
        copies its features from a donor video, and overwrites a background
        span in the current window.
        """
        import random

        num_classes = len(THUMOS14_CLASSES)

        # Find classes that have at least one segment in the index
        candidates = [c for c in THUMOS14_CLASSES if len(self.action_index[c]) > 0]
        if not candidates:
            return features, labels

        # Weight toward rarer classes (inverse frequency)
        counts = [len(self.action_index[c]) for c in candidates]
        max_count = max(counts)
        weights = [max_count / c for c in counts]
        donor_class = random.choices(candidates, weights=weights, k=1)[0]
        cls_idx = CLASS_TO_IDX[donor_class]

        # Pick a random segment from that class
        seg_info = random.choice(self.action_index[donor_class])
        feat_path, seg_c_start, seg_c_end, _, _ = seg_info
        seg_len = seg_c_end - seg_c_start

        if seg_len <= 0 or seg_len > self.window_size // 2:
            return features, labels

        # Load donor features
        donor_feats = torch.load(feat_path, weights_only=True)
        seg_c_end = min(seg_c_end, donor_feats.shape[0])
        donor_clip = donor_feats[seg_c_start:seg_c_end]  # (seg_len, D)
        seg_len = donor_clip.shape[0]

        if seg_len == 0:
            return features, labels

        # Find a background span (all labels zero) in current window
        bg_mask = (labels.sum(dim=1) == 0)  # (W,) — True where background
        W = features.shape[0]

        # Scan for a contiguous background region >= seg_len
        best_start = -1
        run_start = -1
        run_len = 0
        for i in range(W):
            if bg_mask[i]:
                if run_start == -1:
                    run_start = i
                run_len = i - run_start + 1
                if run_len >= seg_len:
                    best_start = run_start
                    break
            else:
                run_start = -1
                run_len = 0

        if best_start == -1:
            return features, labels

        # Paste
        features[best_start : best_start + seg_len] = donor_clip
        labels[best_start : best_start + seg_len, cls_idx] = 1.0

        return features, labels

    def __getitem__(self, idx):
        s = self.samples[idx]
        features = torch.load(s["feat_path"], weights_only=True)  # (N_clips, D)

        # Window jitter during training
        start = s["start"]
        if self.training and self.aug_config.get("window_jitter", False):
            start = self._jitter_start(start, s["n_clips"])

        window = features[start : start + self.window_size]  # (W, D)

        # Build per-clip label vector (multi-label, one-hot)
        num_classes = len(THUMOS14_CLASSES)
        labels = torch.zeros(self.window_size, num_classes)

        clips_per_sec = s["n_clips"] / s["duration"]
        for ann in s["gt_segs"]:
            cls_idx = CLASS_TO_IDX.get(ann["label"], -1)
            if cls_idx == -1:
                continue
            seg_start, seg_end = ann["segment"]
            c_start = int(seg_start * clips_per_sec) - start
            c_end   = int(seg_end   * clips_per_sec) - start
            c_start = max(0, c_start)
            c_end   = min(self.window_size, c_end)
            if c_start < c_end:
                labels[c_start:c_end, cls_idx] = 1.0

        # Cut-and-paste augmentation
        cp_cfg = self.aug_config.get("cutpaste", {})
        if self.training and cp_cfg.get("enabled", False):
            import random
            if random.random() < cp_cfg.get("prob", 0.3):
                window, labels = self._cutpaste(window.clone(), labels, s)

        # Feature-level augmentations (noise, masking, speed, etc.)
        if self.training:
            from src.augmentations import apply_augmentations
            window, labels = apply_augmentations(window, labels, self.aug_config)

        return window, labels