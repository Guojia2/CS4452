"""
src/backbones/videomae2.py

VideoMAE v2 model loading and feature extraction.
Returns dense clip features shaped (num_clips, 768).
"""

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, VideoMAEImageProcessor


MODEL_NAME = "OpenGVLab/VideoMAEv2-Base"


def build_model(device: torch.device):
    original_linspace = torch.linspace

    def patched_linspace(*args, **kwargs):
        if "device" not in kwargs or kwargs["device"] is None:
            kwargs["device"] = torch.device("cpu")
        return original_linspace(*args, **kwargs)

    torch.linspace = patched_linspace
    try:
        config = AutoConfig.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )
        processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            config=config,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        )
    finally:
        torch.linspace = original_linspace

    model.eval().to(device)
    return model, processor


def _to_thwc(video: np.ndarray) -> np.ndarray:
    if video.ndim != 4:
        raise ValueError(f"Expected 4D video array, got shape={video.shape}")

    if video.shape[-1] == 3:
        return video

    if video.shape[0] == 3:
        return np.transpose(video, (1, 2, 3, 0))

    raise ValueError(f"Unsupported video shape {video.shape}")


def _make_clip_starts(num_frames: int, clip_len: int, stride: int):
    if num_frames <= clip_len:
        return [0]

    starts = list(range(0, num_frames - clip_len + 1, stride))
    last_start = num_frames - clip_len
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def extract_features_from_video(
    data: np.ndarray,
    model,
    processor,
    device: torch.device,
    clip_len: int = 16,
    clip_stride: int = 4,
    batch_size: int = 8,
) -> np.ndarray:
    video = _to_thwc(data)

    if video.dtype != np.uint8:
        video = np.clip(video, 0, 255).astype(np.uint8)

    num_frames = video.shape[0]
    starts = _make_clip_starts(num_frames, clip_len, clip_stride)

    all_feats = []

    with torch.no_grad():
        for batch_start in range(0, len(starts), batch_size):
            batch_starts = starts[batch_start:batch_start + batch_size]
            clips = []

            for s in batch_starts:
                clip = video[s:s + clip_len]

                if clip.shape[0] < clip_len:
                    pad_count = clip_len - clip.shape[0]
                    pad_frame = np.repeat(clip[-1:], pad_count, axis=0)
                    clip = np.concatenate([clip, pad_frame], axis=0)

                clips.append([frame for frame in clip])

            inputs = processor(clips, return_tensors="pt")
            pixel_values = inputs["pixel_values"]

            if pixel_values.ndim != 5:
                raise RuntimeError(f"Unexpected pixel_values shape: {pixel_values.shape}")

            pixel_values = pixel_values.permute(0, 2, 1, 3, 4).to(device)

            outputs = model(pixel_values)

            if isinstance(outputs, torch.Tensor):
                hidden = outputs
            elif hasattr(outputs, "last_hidden_state"):
                hidden = outputs.last_hidden_state
            else:
                raise RuntimeError(f"Unexpected model output type: {type(outputs)}")

            if hidden.ndim == 3:
                clip_feats = hidden.mean(dim=1)
            elif hidden.ndim == 2:
                clip_feats = hidden
            else:
                raise RuntimeError(f"Unexpected hidden shape: {hidden.shape}")

            all_feats.append(clip_feats.cpu())

    if not all_feats:
        return np.empty((0, 768), dtype=np.float32)

    features = torch.cat(all_feats, dim=0).numpy().astype(np.float32)
    return features