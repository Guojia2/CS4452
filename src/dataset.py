"""
dataset.py — THUMOS-14 data loading and preprocessing.

Classes
-------
- THUMOSVideoDataset   : Loads raw video clips from S3 on-the-fly.
- THUMOSFeatureDataset : Loads pre-extracted feature tensors from local disk.

Module-level constants
----------------------
- THUMOS14_CLASSES : Dict[int, str] — 1-based index → class name,
                     parsed from data/Class Index.txt at import time.

Manifest format (JSON Lines)
-----------------------------
Each line of a manifest file is a JSON object::

    {"video_key": "videos/train/Archery/v_Archery_g01_c01.avi",
     "label": 3,
     "label_name": "Archery"}

To migrate to a different S3 bucket, pass a new ``S3Store`` instance —
no dataset code needs to change.
"""

from __future__ import annotations

import io
import json
import pathlib
import random
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from src.utils import S3Store, load_class_index

# ---------------------------------------------------------------------------
# Module-level class index — resolved relative to the repository root
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_CLASS_INDEX_PATH = _REPO_ROOT / "data" / "Class Index.txt"

THUMOS14_CLASSES: Dict[int, str] = load_class_index(str(_CLASS_INDEX_PATH))
"""1-based index → class name for all classes in ``data/Class Index.txt``."""

NUM_CLASSES: int = len(THUMOS14_CLASSES)


# ---------------------------------------------------------------------------
# Transform factory
# ---------------------------------------------------------------------------

def build_transforms(split: str = "train", img_size: int = 224) -> T.Compose:
    """Return a ``torchvision.transforms.Compose`` for a given split.

    The transforms operate on a single frame (PIL Image or tensor).
    They are applied frame-by-frame inside ``THUMOSVideoDataset.__getitem__``.

    Parameters
    ----------
    split    : ``"train"`` or ``"val"`` / ``"test"``.
    img_size : Spatial crop/resize target (default 224 for VideoMAE / X3D).
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        return T.Compose([
            T.Resize(int(img_size * 256 / 224)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


# ---------------------------------------------------------------------------
# Internal video decoder
# ---------------------------------------------------------------------------

def _decode_video_bytes(
    video_bytes: bytes,
    clip_len: int,
    frame_interval: int,
    start_idx: Optional[int] = None,
) -> torch.Tensor:
    """Decode raw video bytes and return a clip tensor ``(C, T, H, W)`` as uint8.

    Uses PyAV for container-agnostic decoding (supports .avi, .mp4, etc.).

    Parameters
    ----------
    video_bytes    : Raw bytes of the video file.
    clip_len       : Number of frames to sample.
    frame_interval : Stride between sampled frames (temporal sub-sampling).
    start_idx      : Frame index of the first sampled frame.
                     If None, a random start is chosen.

    Returns
    -------
    Tensor of shape ``(C, T, H, W)`` with dtype ``torch.uint8``.
    """
    try:
        import av
    except ImportError:
        raise ImportError(
            "PyAV is required for video decoding. Install with: pip install av"
        )

    container = av.open(io.BytesIO(video_bytes))
    video_stream = container.streams.video[0]
    video_stream.codec_context.thread_type = av.codec.context.ThreadType.AUTO

    # Decode all frames into a list (memory-friendly with av.seek for large videos)
    frames: List[torch.Tensor] = []
    for frame in container.decode(video=0):
        frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
    container.close()

    total_frames = len(frames)
    span = clip_len * frame_interval

    if total_frames < clip_len:
        # Pad by repeating the last frame
        pad = [frames[-1]] * (clip_len - total_frames)
        frames = frames + pad
        start_idx = 0
    else:
        max_start = max(0, total_frames - span)
        if start_idx is None:
            start_idx = random.randint(0, max_start)
        else:
            start_idx = min(start_idx, max_start)

    sampled = [frames[start_idx + i * frame_interval] for i in range(clip_len)]
    # Each frame: (H, W, C) → stack to (T, H, W, C) → (C, T, H, W)
    clip = torch.stack(sampled, dim=0).permute(3, 0, 1, 2)
    return clip  # uint8, (C, T, H, W)


def _apply_transform_to_clip(
    clip: torch.Tensor,
    transform: Callable,
) -> torch.Tensor:
    """Apply a per-frame spatial transform to a ``(C, T, H, W)`` uint8 tensor.

    Returns a float tensor of the same spatial shape.
    """
    from PIL import Image
    import numpy as np

    c, t, h, w = clip.shape
    # Permute to (T, H, W, C) for PIL conversion
    clip_np = clip.permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
    processed = []
    for i in range(t):
        pil_img = Image.fromarray(clip_np[i])
        processed.append(transform(pil_img))  # → (C, H, W) float tensor
    return torch.stack(processed, dim=1)  # (C, T, H, W)


# ---------------------------------------------------------------------------
# Primary dataset — raw video from S3
# ---------------------------------------------------------------------------

class THUMOSVideoDataset(Dataset):
    """Stream raw videos from S3 and sample clips on-the-fly.

    Parameters
    ----------
    manifest_path : str
        S3 key (relative to ``s3_store``'s prefix) **or** local path to a
        ``.jsonl`` manifest file.
    s3_store      : ``S3Store`` instance.  Pass ``None`` to read manifests and
                    videos from local disk (useful for local testing).
    split         : ``"train"``, ``"val"``, or ``"test"``.
    clip_len      : Number of frames per clip.
    frame_interval: Temporal stride between sampled frames.
    num_clips     : Number of clips to return per video.
                    Train → 1 random clip; val/test → ``num_clips`` uniformly
                    spaced clips for multi-clip inference.
    transform     : Per-frame spatial transform (``torchvision.transforms``).
                    If ``None``, ``build_transforms(split)`` is used.
    img_size      : Spatial resolution passed to ``build_transforms``.
    class_index   : Override class map (default: ``THUMOS14_CLASSES``).
    """

    def __init__(
        self,
        manifest_path: str,
        s3_store: Optional[S3Store],
        split: str = "train",
        clip_len: int = 16,
        frame_interval: int = 4,
        num_clips: int = 1,
        transform: Optional[Callable] = None,
        img_size: int = 224,
        class_index: Optional[Dict[int, str]] = None,
    ) -> None:
        self.s3_store      = s3_store
        self.split         = split
        self.clip_len      = clip_len
        self.frame_interval= frame_interval
        self.num_clips     = num_clips
        self.transform     = transform or build_transforms(split, img_size)
        self.class_index   = class_index or THUMOS14_CLASSES

        self.samples: List[Dict] = self._load_manifest(manifest_path)

    # ------------------------------------------------------------------
    # Manifest loading
    # ------------------------------------------------------------------

    def _load_manifest(self, manifest_path: str) -> List[Dict]:
        """Load manifest from S3 key or local file path."""
        is_local = pathlib.Path(manifest_path).exists()

        if is_local:
            text = pathlib.Path(manifest_path).read_text(encoding="utf-8")
        elif self.s3_store is not None:
            text = self.s3_store.read_text(manifest_path)
        else:
            raise FileNotFoundError(
                f"Manifest not found locally and no S3Store provided: {manifest_path}"
            )

        samples = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                samples.append(json.loads(line))
        return samples

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Return a clip tensor and its integer label.

        Training  (``num_clips == 1``):
            Returns ``(clip, label)`` where ``clip`` is ``(C, T, H, W)``.

        Evaluation (``num_clips > 1``):
            Returns ``(clips, label)`` where ``clips`` is a list of
            ``num_clips`` tensors each of shape ``(C, T, H, W)``.
        """
        sample     = self.samples[idx]
        video_key  = sample["video_key"]
        label      = int(sample["label"])

        video_bytes = self._load_video(video_key)

        if self.num_clips == 1:
            clip = _decode_video_bytes(
                video_bytes, self.clip_len, self.frame_interval, start_idx=None
            )
            clip = _apply_transform_to_clip(clip, self.transform)
            return clip, label
        else:
            # Multi-clip: uniformly space start positions across the video
            clips = self._sample_multi_clips(video_bytes)
            return clips, label

    def _load_video(self, video_key: str) -> bytes:
        """Read video bytes from S3 or local fallback."""
        local_path = pathlib.Path(video_key)
        if local_path.exists():
            return local_path.read_bytes()
        if self.s3_store is not None:
            return self.s3_store.read_bytes(video_key)
        raise FileNotFoundError(
            f"Video not found locally and no S3Store provided: {video_key}"
        )

    def _sample_multi_clips(self, video_bytes: bytes) -> List[torch.Tensor]:
        """Return ``self.num_clips`` uniformly spaced clips from a video."""
        try:
            import av
        except ImportError:
            raise ImportError("PyAV is required. Install with: pip install av")

        container = av.open(io.BytesIO(video_bytes))
        total_frames = container.streams.video[0].frames or 0
        if total_frames == 0:
            # Fallback: decode all to count
            total_frames = sum(1 for _ in container.decode(video=0))
            container.close()
            container = av.open(io.BytesIO(video_bytes))

        container.close()

        span     = self.clip_len * self.frame_interval
        max_start = max(0, total_frames - span)
        starts   = [
            int(max_start * i / max(self.num_clips - 1, 1))
            for i in range(self.num_clips)
        ]

        clips = []
        for start in starts:
            clip = _decode_video_bytes(
                video_bytes, self.clip_len, self.frame_interval, start_idx=start
            )
            clip = _apply_transform_to_clip(clip, self.transform)
            clips.append(clip)
        return clips


# ---------------------------------------------------------------------------
# Secondary dataset — pre-extracted features from local disk
# ---------------------------------------------------------------------------

class THUMOSFeatureDataset(Dataset):
    """Load pre-extracted ``.pt`` feature files from a local directory.

    Useful for fast CPU-side experiments or two-stage pipelines where
    features are extracted offline.

    Directory structure expected::

        feature_root/
            train/
                Archery/
                    v_Archery_g01_c01.pt   # tensor of shape (D,) or (T, D)
            val/
                ...

    Parameters
    ----------
    feature_root : str
        Path to the root directory containing per-split feature files.
    manifest_path : str
        Local ``.jsonl`` manifest (uses ``feature_key`` field instead of
        ``video_key``; falls back to constructing path from class info).
    split        : ``"train"``, ``"val"``, or ``"test"``.
    class_index  : Override class map.
    """

    def __init__(
        self,
        feature_root: str,
        manifest_path: str,
        split: str = "train",
        class_index: Optional[Dict[int, str]] = None,
    ) -> None:
        self.feature_root = pathlib.Path(feature_root)
        self.split        = split
        self.class_index  = class_index or THUMOS14_CLASSES
        self.samples      = self._load_manifest(manifest_path)

    def _load_manifest(self, manifest_path: str) -> List[Dict]:
        text = pathlib.Path(manifest_path).read_text(encoding="utf-8")
        return [json.loads(l) for l in text.splitlines() if l.strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        label  = int(sample["label"])

        # Prefer explicit feature_key; fall back to constructing from video_key
        if "feature_key" in sample:
            feat_path = self.feature_root / sample["feature_key"]
        else:
            stem = pathlib.Path(sample["video_key"]).stem
            class_name = sample.get("label_name", self.class_index.get(label + 1, str(label)))
            feat_path = self.feature_root / self.split / class_name / f"{stem}.pt"

        features = torch.load(feat_path, map_location="cpu")
        return features, label


# ---------------------------------------------------------------------------
# Collate helper for multi-clip evaluation batches
# ---------------------------------------------------------------------------

def multi_clip_collate(batch):
    """Custom collate for multi-clip mode.

    Each item is ``(List[Tensor], int)``.
    Returns ``(clips_batch, labels)`` where ``clips_batch`` is a list of
    ``(B, C, T, H, W)`` tensors, one per clip position.
    """
    clips_list, labels = zip(*batch)
    num_clips = len(clips_list[0])
    batched_clips = [
        torch.stack([clips[i] for clips in clips_list], dim=0)
        for i in range(num_clips)
    ]
    labels = torch.tensor(labels, dtype=torch.long)
    return batched_clips, labels
