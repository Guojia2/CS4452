"""
modal_pipeline/modal_extract.py

Modal pipeline for extracting VideoMAE v2 features.
"""

import os
import modal


APP_NAME = "thumos-videomae2-feature-extraction"
VOLUME_NAME = "Thumos14"
VOLUME_MOUNT = "/vol"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy",
        "tqdm",
        "transformers==4.30.2",
        "einops",
        "easydict",
        "timm",
        "Pillow",
    )
    .add_local_dir(".", remote_path="/root/project", copy=True)
)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    gpu="A10G",
    timeout=86400,
)
def extract_split(
    split: str,
    clip_len: int,
    clip_stride: int,
    batch_size: int,
    shard_idx: int = 0,
    num_shards: int = 1,
):
    import numpy as np
    import torch
    import tqdm
    import sys

    sys.path.append("/root/project")

    from libs.backbones.videomae2 import build_model, extract_features_from_video

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Extracting VideoMAE v2 features for split={split} on device={device} "
        f"clip_len={clip_len} clip_stride={clip_stride} batch_size={batch_size} "
        f"shard={shard_idx}/{num_shards}"
    )

    model, processor = build_model(device)

    input_dir = os.path.join(VOLUME_MOUNT, f"npy/{split}_npy")
    output_dir = os.path.join(VOLUME_MOUNT, f"features/videomae2-new")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_dir):
        print(f"Directory does not exist: {input_dir}")
        return

    npy_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".npy")])
    npy_files = npy_files[shard_idx::num_shards]

    if not npy_files:
        print(f"No .npy files found in {input_dir} for shard {shard_idx}/{num_shards}")
        return

    total = len(npy_files)
    saved = 0
    skipped = 0
    failed = 0

    for filename in tqdm.tqdm(npy_files):
        video_name = os.path.splitext(filename)[0]
        out_path = os.path.join(output_dir, f"{video_name}.npy")

        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            in_path = os.path.join(input_dir, filename)
            data = np.load(in_path)

            features = extract_features_from_video(
                data=data,
                model=model,
                processor=processor,
                device=device,
                clip_len=clip_len,
                clip_stride=clip_stride,
                batch_size=batch_size,
            )

            np.save(out_path, features)
            saved += 1

            if saved <= 3:
                print(f"Saved {video_name}.npy with shape {features.shape}")

        except Exception as e:
            failed += 1
            print(f"Failed on {filename}: {e}")

    print(
        f"Finished split={split}. shard={shard_idx}/{num_shards} "
        f"total={total} saved={saved} skipped={skipped} failed={failed}"
    )
    print(
        f"ActionFormer config should start with: "
        f"input_dim=768, num_frames={clip_len}, feat_stride={clip_stride}"
    )

    volume.commit()


@app.local_entrypoint()
def main(
    clip_len: int = 16,
    clip_stride: int = 8,
    batch_size: int = 8,
):
    print(
        "Running VideoMAE v2 feature extraction on Modal "
        f"with clip_len={clip_len}, clip_stride={clip_stride}, batch_size={batch_size}"
    )

    jobs = []

    for split in ["validation", "test"]:
        for shard_idx in range(4):
            jobs.append((split, clip_len, clip_stride, batch_size, shard_idx, 4))

    list(extract_split.starmap(jobs))

    print("Done.")