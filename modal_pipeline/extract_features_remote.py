"""
Feature extraction on Modal — parallel across all videos.

Reads raw videos from the shared dataset volume (main env).
Writes (N_clips, 2048) .pt feature files to the work volume (alex-dev env).

Run:
  modal run modal_pipeline/extract_features_remote.py
"""

import modal
import os

from modal_pipeline.app import app, image, dataset_volume, work_volume, DATASET_PATH, WORK_PATH

# ── GPU worker — X3D-M loaded once per container via @modal.enter() ──────────

@app.cls(
    image=image,
    gpu="t4",                   # T4 sufficient for X3D-M inference
    volumes={
        DATASET_PATH: dataset_volume,   # read-only: raw videos + annotations
        WORK_PATH:    work_volume,       # read-write: output features
    },
    timeout=600,                # 10-min ceiling per video call
    cpu=4,                      # 2 for decord decode threads + 1 prefetch + 1 spare
    memory=8192,
    max_containers=5,        # stay under Modal free-tier GPU cap
)
class FeatureExtractor:

    @modal.enter()
    def load_model(self):
        import sys
        import torch
        sys.path.insert(0, "/root")

        # Cache X3D-M weights to the work volume — downloaded once, reused forever
        torch.hub.set_dir(f"{WORK_PATH}/torch_hub")

        from src.models.backbone import build_backbone
        from src.extract_features import build_transforms

        self.device = torch.device("cuda")
        self.backbone, _ = build_backbone("x3d_m", pretrained=True)
        self.backbone = self.backbone.to(self.device).eval()
        self.transform = build_transforms()

    @modal.method()
    def extract(self, video_name: str) -> str:
        import sys
        import torch
        import os
        sys.path.insert(0, "/root")

        from src.extract_features import extract_video

        feat_path = f"{WORK_PATH}/features/clip_level/{video_name}.pt"
        if os.path.exists(feat_path):
            return f"skip:{video_name}"

        if "validation" in video_name:
            video_path = f"{DATASET_PATH}/raw/videos/val/{video_name}.mp4"
        else:
            video_path = f"{DATASET_PATH}/raw/videos/test/{video_name}.mp4"

        if not os.path.exists(video_path):
            return f"missing:{video_name}"

        os.makedirs(f"{WORK_PATH}/features/clip_level", exist_ok=True)

        feats = extract_video(
            video_path, self.backbone, self.device,
            clip_len_sec=2.0, stride_sec=1.0, num_frames=16,
            transform=self.transform, batch_size=64,
        )

        if feats is None:
            return f"failed:{video_name}"

        torch.save(feats, feat_path)
        work_volume.commit()
        return f"done:{video_name}:{list(feats.shape)}"


# ── Local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    import json

    ann_path = "data/THUMOS14/annotations/annotations/thumos_14_anno.json"
    if not os.path.exists(ann_path):
        # Fallback in case local data isn't present — list from volume
        print("Local annotations not found. Reading video list from Modal volume...")
        import subprocess
        result = subprocess.run(
            ["modal", "volume", "ls", "--env", "main", "thumos-vol", "raw/videos/val"],
            capture_output=True, text=True,
        )
        val_names = [
            l.split("/")[-1].replace(".mp4", "")
            for l in result.stdout.splitlines() if l.endswith(".mp4")
        ]
        result = subprocess.run(
            ["modal", "volume", "ls", "--env", "main", "thumos-vol", "raw/videos/test"],
            capture_output=True, text=True,
        )
        test_names = [
            l.split("/")[-1].replace(".mp4", "")
            for l in result.stdout.splitlines() if l.endswith(".mp4")
        ]
        video_names = sorted(val_names + test_names)
    else:
        with open(ann_path) as f:
            db = json.load(f)["database"]
        video_names = sorted(db.keys())

    print(f"Submitting extraction for {len(video_names)} videos...")

    extractor = FeatureExtractor()
    done = skipped = failed = missing = 0

    for i, result in enumerate(
        extractor.extract.map(video_names, order_outputs=False), start=1
    ):
        tag = result.split(":")[0]
        if tag == "done":      done += 1
        elif tag == "skip":    skipped += 1
        elif tag == "missing": missing += 1
        else:                  failed += 1
        print(f"[{i:3d}/{len(video_names)}] {result}")

    print(
        f"\nDone — extracted: {done}  skipped: {skipped}  "
        f"missing: {missing}  failed: {failed}"
    )
