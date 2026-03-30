"""
Feature extraction for THUMOS-14 — optimized for single-GPU local use.

Key design choices vs. the original DataLoader-based approach:
- Process one video at a time: decode ALL frames in a single sequential av pass
  (no per-clip seeking → eliminates H.264 keyframe seek overhead)
- Batch backbone inference with torch.autocast FP16 for ~2x GPU throughput
- Skip videos whose .pt file already exists (resume after interruption)
- 2-worker multiprocessing: each worker gets its own GPU context and processes
  a disjoint subset of videos → parallelises CPU decode with GPU inference
"""

from __future__ import annotations

import os
import math
import time
from typing import Dict, List, Optional, Tuple

import av
import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from PIL import Image

import torchvision.transforms as T

from src.models.backbone import build_backbone
from src.utils import get_logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_transforms(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize(int(img_size * 256 / 224)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def collect_video_list(video_dirs: List[str], ann_path: str) -> List[Tuple[str, str]]:
    """
    Return list of (video_name, video_path) for every video that exists on disk.
    Scans all provided video_dirs.
    """
    import json
    with open(ann_path) as f:
        db = json.load(f)["database"]
    all_names = list(db.keys())

    results = []
    for name in all_names:
        for vdir in video_dirs:
            p = os.path.join(vdir, f"{name}.mp4")
            if os.path.exists(p):
                results.append((name, p))
                break
    return results


# ---------------------------------------------------------------------------
# Per-video extraction
# ---------------------------------------------------------------------------

def _build_clip_index(
    total_frames: int, fps: float, clip_len_sec: float, stride_sec: float, num_frames: int
) -> List[np.ndarray]:
    """Return list of frame-index arrays, one per clip."""
    clip_frames   = max(1, int(round(clip_len_sec * fps)))
    stride_frames = max(1, int(round(stride_sec * fps)))
    clips = []
    start = 0
    while start + clip_frames <= total_frames:
        idx = np.linspace(start, start + clip_frames - 1, num_frames, dtype=np.int64)
        clips.append(idx)
        start += stride_frames
    return clips


def _clips_to_tensor(
    frames_nhwc: np.ndarray,        # (N_unique_frames, H, W, C) uint8
    frame_lookup: dict,             # global_frame_idx → position in frames_nhwc
    clip_indices: List[np.ndarray], # list of per-clip global frame index arrays
    transform: T.Compose,
) -> torch.Tensor:
    """Convert a batch of clips to a (B, C, T, H, W) float tensor on CPU."""
    clips = []
    for idx_arr in clip_indices:
        local = [frame_lookup[i] for i in idx_arr]
        frames = [transform(Image.fromarray(frames_nhwc[l])) for l in local]
        clips.append(torch.stack(frames).permute(1, 0, 2, 3))  # (C, T, H, W)
    return torch.stack(clips)  # (B, C, T, H, W)


def extract_video(
    video_path: str,
    backbone: torch.nn.Module,
    device: torch.device,
    clip_len_sec: float,
    stride_sec: float,
    num_frames: int,
    transform: T.Compose,
    batch_size: int = 64,
) -> Optional[torch.Tensor]:
    """
    Decode a video and extract backbone features.

    Uses decord for fast batch frame fetching when available, falls back to av.
    A prefetch thread prepares the next CPU batch while the GPU processes the
    current one, keeping both CPU and GPU busy throughout.

    Returns tensor of shape (N_clips, feat_dim), or None on failure.
    """
    import threading
    import queue

    # ── 1. Open video and build clip index ───────────────────────────────────
    try:
        import decord
        reader = decord.VideoReader(video_path, ctx=decord.cpu(), num_threads=2)
        fps          = reader.get_avg_fps()
        total_frames = len(reader)
        use_decord   = True
    except Exception:
        use_decord = False
        try:
            container = av.open(video_path)
        except Exception:
            return None
        stream       = container.streams.video[0]
        fps          = float(stream.average_rate)
        total_frames = stream.frames or 0

    if fps <= 0 or total_frames == 0:
        if not use_decord:
            container.close()
        return None

    all_clip_indices = _build_clip_index(
        total_frames, fps, clip_len_sec, stride_sec, num_frames
    )
    if not all_clip_indices:
        if not use_decord:
            container.close()
        return None

    # ── 2. Decode frames ─────────────────────────────────────────────────────
    if use_decord:
        # Collect every unique frame index needed across all clips
        unique_idx = sorted(set(int(i) for c in all_clip_indices for i in c))
        unique_idx = [min(i, total_frames - 1) for i in unique_idx]
        frames_t   = reader.get_batch(unique_idx)           # (N, H, W, C) decord NDArray
        frames_np  = frames_t.asnumpy()                    # -> numpy uint8
        frame_lookup = {g: l for l, g in enumerate(unique_idx)}
    else:
        # Fallback: sequential av decode (no seeking)
        raw: List[np.ndarray] = []
        for frame in container.decode(video=0):
            raw.append(frame.to_ndarray(format="rgb24"))
        container.close()
        if not raw:
            return None
        total_frames = len(raw)
        # Recompute clip index with actual frame count
        all_clip_indices = _build_clip_index(
            total_frames, fps, clip_len_sec, stride_sec, num_frames
        )
        if not all_clip_indices:
            return None
        unique_idx   = sorted(set(int(i) for c in all_clip_indices for i in c))
        frames_np    = np.stack([raw[min(i, total_frames - 1)] for i in unique_idx])
        frame_lookup = {g: l for l, g in enumerate(unique_idx)}
        del raw

    # ── 3. Pipelined CPU→GPU processing ─────────────────────────────────────
    # Producer thread: build CPU batches and put them into a queue.
    # Main thread: pull from queue, run GPU inference.
    # This keeps CPU and GPU busy simultaneously.

    prefetch_q: queue.Queue = queue.Queue(maxsize=2)

    def _producer():
        for i in range(0, len(all_clip_indices), batch_size):
            chunk = all_clip_indices[i : i + batch_size]
            batch_cpu = _clips_to_tensor(frames_np, frame_lookup, chunk, transform)
            prefetch_q.put(batch_cpu)
        prefetch_q.put(None)  # sentinel

    producer = threading.Thread(target=_producer, daemon=True)
    producer.start()

    feature_chunks: List[torch.Tensor] = []
    while True:
        batch_cpu = prefetch_q.get()
        if batch_cpu is None:
            break
        batch_gpu = batch_cpu.to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            feats = backbone(batch_gpu)
        feature_chunks.append(feats.cpu())

    producer.join()
    del frames_np

    return torch.cat(feature_chunks, dim=0)  # (N_clips, feat_dim)


# ---------------------------------------------------------------------------
# Worker entry point (used by multiprocessing)
# ---------------------------------------------------------------------------

def _worker(
    worker_id: int,
    video_list: List[Tuple[str, str]],
    feat_dir: str,
    backbone_name: str,
    clip_len_sec: float,
    stride_sec: float,
    num_frames: int,
    img_size: int,
    batch_size: int,
    log_dir: Optional[str],
):
    logger = get_logger(f"extract_worker_{worker_id}", log_dir=log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Worker {worker_id}: device={device}, {len(video_list)} videos assigned")

    backbone, _ = build_backbone(backbone_name, pretrained=True)
    backbone = backbone.to(device).eval()
    transform = build_transforms(img_size=img_size)

    os.makedirs(feat_dir, exist_ok=True)

    done = 0
    skipped = 0
    t0 = time.time()

    for video_name, video_path in video_list:
        out_path = os.path.join(feat_dir, f"{video_name}.pt")
        if os.path.exists(out_path):
            skipped += 1
            continue

        feats = extract_video(
            video_path, backbone, device,
            clip_len_sec, stride_sec, num_frames,
            transform, batch_size=batch_size,
        )

        if feats is None:
            logger.warning(f"Worker {worker_id}: skipping {video_name} (no frames decoded)")
            continue

        torch.save(feats, out_path)
        done += 1
        elapsed = time.time() - t0
        per_video = elapsed / done
        remaining = len(video_list) - done - skipped
        eta_min = remaining * per_video / 60
        logger.info(
            f"Worker {worker_id}: [{done}/{len(video_list)-skipped}] {video_name} "
            f"-> {feats.shape}  {per_video:.1f}s/video  ETA {eta_min:.1f} min"
        )

    logger.info(
        f"Worker {worker_id}: done. Extracted {done} videos, skipped {skipped} already done."
    )


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def run_feature_extraction(config_path: str) -> None:
    config = load_config(config_path)

    backbone_name = config["model"]["backbone"]
    clip_len_sec  = config["data"]["clip_len_sec"]
    stride_sec    = config["data"]["stride_sec"]
    num_frames    = config["data"]["num_frames"]
    img_size      = config["data"].get("img_size", 224)

    data_root = config["paths"]["data_root"]
    feat_dir  = config["paths"]["feature_dir"]
    log_dir   = config["paths"].get("log_dir")

    ext_cfg   = config.get("extraction", {})
    num_workers  = ext_cfg.get("num_workers", 1)
    batch_size   = ext_cfg.get("batch_size", 64)

    ann_path = config["paths"].get(
        "ann_path",
        os.path.join(data_root, "annotations", "thumos_14_anno.json"),
    )

    # Collect all videos from both val and test directories
    video_dirs = [
        os.path.join(data_root, "videos", "val"),
        os.path.join(data_root, "videos", "test"),
    ]
    video_list = collect_video_list(video_dirs, ann_path)

    logger = get_logger("extract_main", log_dir=log_dir)
    logger.info(f"Found {len(video_list)} videos across val+test")
    already_done = sum(
        1 for name, _ in video_list
        if os.path.exists(os.path.join(feat_dir, f"{name}.pt"))
    )
    logger.info(f"Already extracted: {already_done} / {len(video_list)}")

    if num_workers <= 1:
        # Single-process path (simpler, easier to debug)
        _worker(
            worker_id=0,
            video_list=video_list,
            feat_dir=feat_dir,
            backbone_name=backbone_name,
            clip_len_sec=clip_len_sec,
            stride_sec=stride_sec,
            num_frames=num_frames,
            img_size=img_size,
            batch_size=batch_size,
            log_dir=log_dir,
        )
    else:
        # Multiprocessing: split video list across workers
        # Filter out already-done videos before splitting to balance load
        remaining = [(n, p) for n, p in video_list
                     if not os.path.exists(os.path.join(feat_dir, f"{n}.pt"))]
        chunk_size = math.ceil(len(remaining) / num_workers)
        chunks = [remaining[i : i + chunk_size] for i in range(0, len(remaining), chunk_size)]

        logger.info(f"Spawning {len(chunks)} workers for {len(remaining)} remaining videos")

        mp.set_start_method("spawn", force=True)
        processes = []
        for wid, chunk in enumerate(chunks):
            p = mp.Process(
                target=_worker,
                args=(wid, chunk, feat_dir, backbone_name,
                      clip_len_sec, stride_sec, num_frames,
                      img_size, batch_size, log_dir),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        logger.info("All workers finished.")
