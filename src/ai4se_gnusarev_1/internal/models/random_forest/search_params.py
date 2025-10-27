import argparse
from pathlib import Path

import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import yaml
import polars as pl


def main():
    args = parse_args()
    args.func(args)


def parse_args():
    prepare_data_parser = argparse.ArgumentParser()
    prepare_data_parser.set_defaults(func=search_params)
    prepare_data_parser.add_argument(
        "--cfg_path",
        help="Path to model config",
        type=Path,
    )
    prepare_data_parser.add_argument(
        "--n_trials",
        help="Number of optuna trials",
        type=int,
        default=10,
    )

    return prepare_data_parser.parse_args()


def search_params(args):
    objective = Objective(args.cfg_path)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print(f"Best parameters: {study.best_params}")
    print(f"Best accuracy: {study.best_value}")


class Objective:
    def __init__(
        self,
        cfg_path: str,
    ):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.parameters = cfg["parameters"]
        train = pl.read_excel(cfg["train_path"])
        self.X, self.y = (
            train.get_column("message"),
            train.get_column("is_toxic"),
        )

    def __call__(self, trial: optuna.Trial):
        # parameters
        parameters = self.parameters["static"]
        for param_name, values in self.parameters["dynamic"].items():
            parameters[param_name] = getattr(trial, values["method"])(
                name=param_name, **values["range"]
            )
        pipeline = make_pipeline(
            TfidfVectorizer(), RandomForestClassifier(**parameters)
        )
        score = cross_val_score(pipeline, self.X, self.y, n_jobs=-1, cv=10)
        accuracy = score.mean()
        return accuracy


if __name__ == "__main__":
    main()
