"""Local runner — feature extraction, training, and evaluation."""

import argparse

CONFIG = "configs/local_config.yaml"


def cmd_extract(args):
    from src.extract_features import run_feature_extraction
    run_feature_extraction(args.config)


def cmd_train(args):
    import yaml
    from src.train import train
    with open(args.config) as f:
        config = yaml.safe_load(f)
    train(config)


def cmd_evaluate(args):
    import yaml
    from src.evaluate import evaluate
    with open(args.config) as f:
        config = yaml.safe_load(f)
    evaluate(config, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("extract")

    p_train = sub.add_parser("train")

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--checkpoint", required=True)

    args = parser.parse_args()

    if args.cmd == "extract":
        cmd_extract(args)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "evaluate":
        cmd_evaluate(args)
