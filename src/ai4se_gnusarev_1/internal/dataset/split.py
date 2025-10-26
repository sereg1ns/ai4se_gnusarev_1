import typing as tp

import numpy as np
import polars as pl

def split_df(
        df: pl.DataFrame,
        split_fraction: float = 0.8
    ) -> tp.Tuple[pl.DataFrame, pl.DataFrame]:
    distribution = np.random.binomial(1, split_fraction, size=df.shape[0])
    big = df.filter((distribution == 1))
    small = df.filter((distribution == 0))
    return big, small

def check_balance(df: pl.DataFrame) -> float:
    ones = df.filter(pl.col("is_toxic") == 1).shape[0]
    return ones / df.shape[0]
