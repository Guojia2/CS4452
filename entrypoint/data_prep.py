"""
entrypoint/data_prep.py — Upload raw THUMOS-14 videos to S3 and write manifests.

Usage
-----
python entrypoint/data_prep.py \\
    --local-root /path/to/thumos14 \\
    --bucket     thumos14-videos   \\
    --region     us-east-1         \\
    [--prefix    ""]               \\
    [--splits    train val test]   \\
    [--dry-run]

Expected local directory structure
-----------------------------------
<local-root>/
    train/
        Archery/
            v_Archery_g01_c01.avi
            ...
        Basketball/
            ...
    val/
        ...
    test/
        ...

S3 output layout
----------------
<prefix>/videos/<split>/<class>/<filename>
<prefix>/manifests/<split>.jsonl

Manifest line format
--------------------
{"video_key": "videos/train/Archery/v_Archery_g01_c01.avi",
 "label": 3,
 "label_name": "Archery"}

Note: ``label`` is the 0-based index used for training tensors
      (i.e., ``<1-based index from Class Index.txt> - 1``).

Bucket migration
----------------
Pass --bucket <new-bucket>.  The S3Store abstraction handles the rest.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

# Ensure repo root is on path so src can be imported
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.utils import S3Store, get_logger, load_class_index

logger = get_logger("data_prep")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_class_name_to_idx(class_index: dict) -> dict:
    """Invert {1-based idx: name} → {name: 0-based idx}."""
    return {name: (idx - 1) for idx, name in class_index.items()}


def _supported_ext(filename: str) -> bool:
    return pathlib.Path(filename).suffix.lower() in (".avi", ".mp4", ".mov", ".mkv")


# ---------------------------------------------------------------------------
# Core upload + manifest
# ---------------------------------------------------------------------------

def upload_split(
    local_split_dir: pathlib.Path,
    split: str,
    s3_store: S3Store,
    video_prefix: str,
    class_name_to_idx: dict,
    dry_run: bool = False,
) -> list[dict]:
    """Upload all videos in a split directory and return manifest records."""
    records = []

    class_dirs = sorted(d for d in local_split_dir.iterdir() if d.is_dir())
    if not class_dirs:
        logger.warning(f"No class subdirectories found in {local_split_dir}")
        return records

    for class_dir in class_dirs:
        class_name = class_dir.name

        if class_name not in class_name_to_idx:
            logger.warning(f"  Class '{class_name}' not in Class Index — skipping")
            continue

        label = class_name_to_idx[class_name]   # 0-based

        video_files = sorted(
            f for f in class_dir.iterdir()
            if f.is_file() and _supported_ext(f.name)
        )

        for video_path in video_files:
            s3_key = f"{video_prefix}/{split}/{class_name}/{video_path.name}"

            if not dry_run:
                logger.info(f"  Uploading {video_path.name} → s3://{s3_store.bucket}/{s3_key}")
                s3_store.upload_file(str(video_path), s3_key)
            else:
                logger.info(f"  [DRY RUN] Would upload {video_path} → {s3_key}")

            records.append({
                "video_key":  s3_key,
                "label":      label,
                "label_name": class_name,
                "split":      split,
            })

    return records


def write_manifest(
    records: list[dict],
    split: str,
    s3_store: S3Store,
    manifest_prefix: str,
    dry_run: bool = False,
) -> None:
    """Write a JSON Lines manifest for a split to S3."""
    manifest_key = f"{manifest_prefix}/{split}.jsonl"
    lines = "\n".join(json.dumps(r) for r in records)

    if not dry_run:
        logger.info(f"Writing manifest ({len(records)} records) → s3://{s3_store.bucket}/{manifest_key}")
        s3_store.write_text(manifest_key, lines)
    else:
        logger.info(f"[DRY RUN] Would write manifest to {manifest_key} ({len(records)} records)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload THUMOS-14 videos to S3 and write manifests"
    )
    parser.add_argument(
        "--local-root", required=True,
        help="Root directory containing per-split class subdirectories"
    )
    parser.add_argument(
        "--bucket", required=True,
        help="Destination S3 bucket name"
    )
    parser.add_argument(
        "--region", default="us-east-1",
        help="AWS region (default: us-east-1)"
    )
    parser.add_argument(
        "--prefix", default="",
        help="Key prefix inside the bucket (default: empty)"
    )
    parser.add_argument(
        "--video-prefix", default="videos",
        help="Prefix for video objects (default: 'videos')"
    )
    parser.add_argument(
        "--manifest-prefix", default="manifests",
        help="Prefix for manifest objects (default: 'manifests')"
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        help="Splits to process (default: train val test)"
    )
    parser.add_argument(
        "--class-index", default=str(_REPO_ROOT / "data" / "Class Index.txt"),
        help="Path to Class Index.txt"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without uploading"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    class_index      = load_class_index(args.class_index)
    class_name_to_idx = _build_class_name_to_idx(class_index)

    logger.info(f"Loaded {len(class_index)} classes from {args.class_index}")

    s3_store = S3Store(
        bucket  = args.bucket,
        region  = args.region,
        prefix  = args.prefix,
    )

    local_root = pathlib.Path(args.local_root)

    for split in args.splits:
        split_dir = local_root / split
        if not split_dir.exists():
            logger.warning(f"Split directory not found, skipping: {split_dir}")
            continue

        logger.info(f"Processing split: {split}")
        records = upload_split(
            local_split_dir   = split_dir,
            split             = split,
            s3_store          = s3_store,
            video_prefix      = args.video_prefix,
            class_name_to_idx = class_name_to_idx,
            dry_run           = args.dry_run,
        )

        write_manifest(
            records         = records,
            split           = split,
            s3_store        = s3_store,
            manifest_prefix = args.manifest_prefix,
            dry_run         = args.dry_run,
        )

        logger.info(f"Split '{split}' done: {len(records)} videos")

    logger.info("Data prep complete.")


if __name__ == "__main__":
    main()
