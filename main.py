"""
main.py — Local CLI entrypoint to dispatch Modal training and evaluation.

Usage
-----
# Launch training on Modal (uses configs/base_config.yaml)
python main.py train

# Launch training with config overrides
python main.py train --override model.backbone=x3d_m training.epochs=5

# Launch evaluation on Modal
python main.py eval --checkpoint /vol/checkpoints/best.pt

# Evaluate on val split with 10 test clips
python main.py eval --checkpoint /vol/checkpoints/best.pt --split val --num-clips 10
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dispatch THUMOS-14 training / evaluation to Modal"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ------------------------------------------------------------------ train
    train_parser = subparsers.add_parser("train", help="Launch training on Modal A10G")
    train_parser.add_argument(
        "--config", default="configs/base_config.yaml",
        help="Local path to YAML config (embedded in Modal image)"
    )
    train_parser.add_argument(
        "--override", nargs="*", default=[], metavar="KEY=VALUE",
        help="Config overrides, e.g. model.backbone=x3d_m training.epochs=5"
    )

    # ------------------------------------------------------------------ eval
    eval_parser = subparsers.add_parser("eval", help="Launch evaluation on Modal T4")
    eval_parser.add_argument(
        "--checkpoint", default="/vol/checkpoints/best.pt",
        help="Path to checkpoint inside the Modal Volume"
    )
    eval_parser.add_argument(
        "--split", default="test", choices=["val", "test"],
        help="Dataset split to evaluate"
    )
    eval_parser.add_argument(
        "--num-clips", type=int, default=5,
        help="Number of clips per video for multi-clip inference"
    )
    eval_parser.add_argument(
        "--config", default="configs/base_config.yaml",
        help="Local path to YAML config"
    )
    eval_parser.add_argument(
        "--override", nargs="*", default=[], metavar="KEY=VALUE",
        help="Config overrides"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = args.override or []

    # Import Modal functions lazily so `python main.py --help` works without Modal installed
    from modal_pipeline.app import run_training, run_eval

    if args.mode == "train":
        print(f"[main.py] Dispatching training to Modal  overrides={overrides}")
        result = run_training.remote(overrides=overrides or None)
        print(f"[main.py] Training complete. Best val top-1: {result:.2f}%")

    elif args.mode == "eval":
        print(
            f"[main.py] Dispatching evaluation to Modal  "
            f"checkpoint={args.checkpoint}  split={args.split}  num_clips={args.num_clips}"
        )
        results = run_eval.remote(
            checkpoint_path = args.checkpoint,
            split           = args.split,
            num_test_clips  = args.num_clips,
            overrides       = overrides or None,
        )
        print(f"\nmAP  = {results['mAP']:.4f}")
        print(f"Top-1= {results['top1']:.2f}%")
        print(f"Top-5= {results['top5']:.2f}%")


if __name__ == "__main__":
    main()
