import argparse
from pathlib import Path
import os

import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
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

    return prepare_data_parser.parse_args()

def search_params(args):
    with open(args.cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    pipeline = make_pipeline(
        TfidfVectorizer(),
        RandomForestClassifier(**cfg["parameters"])
    )
    param_distributions = {
        'C': optuna.distributions.LogUniformDistribution(1e-10, 1e+10)
    }
    optuna_search = optuna.integration.OptunaSearchCV(
        clf,
        param_distributions
    )
    train = pl.read_excel(cfg["data"]["train_path"])
    X, y = train.get_column("message"), train.get_column("is_toxic")
    optuna_search.fit(X, y)

if __name__ == "__main__":
    main()
