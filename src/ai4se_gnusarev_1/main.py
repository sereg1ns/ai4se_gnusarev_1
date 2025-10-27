import argparse
from pathlib import Path
import os

from ai4se_gnusarev_1.consts import MODELS


def main():
    args = parse_args()
    args.func(args)


def parse_args():
    prepare_data_parser = argparse.ArgumentParser()
    prepare_data_parser.set_defaults(func=train_model)
    default_model_path = Path(
        "/mnt/c/Users/SGnusarev/Desktop/VSCode/ai4se_gnusarev_1/results"
    )
    prepare_data_parser.add_argument(
        "--model_type",
        help="Model type to train",
        type=str,
    )
    prepare_data_parser.add_argument(
        "--cfg_path",
        help="Path to model config",
        type=Path,
    )
    prepare_data_parser.add_argument(
        "-o",
        "--output",
        help="Path to save prepared split dataset to",
        type=Path,
        default=default_model_path,
    )

    return prepare_data_parser.parse_args()


def train_model(args):
    train = MODELS.get(args.model_type)
    if train is None:
        raise ValueError("Wrong model type!")
    # train
    best_metric = train(
        args.cfg_path, os.path.join(args.output, args.model_type)
    )
    print(f"Best model metric: {best_metric}")


if __name__ == "__main__":
    main()
