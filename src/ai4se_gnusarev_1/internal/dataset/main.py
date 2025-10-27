import argparse
from pathlib import Path
import os

import polars as pl

from ai4se_gnusarev_1.internal.dataset.preprocess import Preprocess
from ai4se_gnusarev_1.internal.dataset.split import split_df, check_balance


DATA_PREFIX = "prepared_data"


def main():
    args = parse_args()
    args.func(args)


def parse_args():
    prepare_data_parser = argparse.ArgumentParser()
    default_data_path = Path("")
    prepare_data_parser.set_defaults(func=prepare_data)
    prepare_data_parser.add_argument(
        "--filters",
        help="Delimeted list of filters to apply",
        type=str,
        default="remove_url,expand_contraction,remove_special_chars,remove_repetitions,replace_curse_rephrasing",
    )
    prepare_data_parser.add_argument(
        "--lower",
        help="Whether to lowercase the message",
        type=bool,
        default=True,
    )
    prepare_data_parser.add_argument(
        "--input",
        help="Path to load raw dataset",
        type=Path,
    )
    prepare_data_parser.add_argument(
        "-o",
        "--output",
        help="Path to save prepared split dataset to",
        type=Path,
        default=default_data_path,
    )

    return prepare_data_parser.parse_args()


def prepare_data(args):
    # process
    prepared_filters = (
        args.filters.replace(" ", "").replace("\n", "").replace("\t", "")
    )
    preprocessor = Preprocess(
        filters=[f for f in prepared_filters.split(",")],
        lower=args.lower,
    )
    processed_data: pl.DataFrame = preprocessor(args.input)
    print(f"1 balance: {check_balance(processed_data):.2f}")
    # split
    train, test = split_df(processed_data)
    ## check
    print(f"1 balance in test: {check_balance(test):.2f}")
    print(f"1 balance in train: {check_balance(train):.2f}")
    # save
    test.write_excel(os.path.join(args.output, DATA_PREFIX + "_test.xlsx"))
    train.write_excel(os.path.join(args.output, DATA_PREFIX + "_train.xlsx"))


if __name__ == "__main__":
    main()
