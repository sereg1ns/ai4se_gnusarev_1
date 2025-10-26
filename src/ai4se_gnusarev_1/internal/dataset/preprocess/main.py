import re
import typing as tp

import polars as pl

from ai4se_gnusarev_1.internal.dataset.preprocess.consts import (
    URL_REGEX,
    CONTRACTION_MAPPING,
    CURSE_REPHRASING,
)


class Preprocess:
    def __init__(
        self,
        filters: tp.List[str],
        lower: bool = True,
    ):
        self.filters = filters
        self.lower = lower

    @staticmethod
    def remove_url(text: str) -> str:
        return URL_REGEX.sub(" ", text)

    @staticmethod
    def expand_contraction(text: str):
        specials = ["’", "‘", "´", "`", "'"]
        for s in specials:
            text = text.replace(s, "'")
            text = " ".join(
                [
                    CONTRACTION_MAPPING[t] if t in CONTRACTION_MAPPING else t
                    for t in text.split(" ")
                ]
            )
        return text

    @staticmethod
    def remove_special_chars(text: str) -> str:
        pattern = r"[^a-z0-9!@#\$%\^\*\+\?\&\_\-,\.' ]"
        return text.replace(pattern, " ")

    @staticmethod
    def remove_repetitions(text: str) -> str:
        pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
        return pattern.sub(r"\1", text)

    @staticmethod
    def replace_curse_rephrasing(text: str) -> str:
        for target, patterns in CURSE_REPHRASING.items():
            for pat in patterns:
                text = re.sub(pat, target, text)
        text = re.sub(r"[^a-z' ]", " ", text)
        return text

    def _combined_filter(self, text: str) -> str:
        for filter in self.filters:
            text = getattr(self, filter)(text)
        return text

    def __call__(self, path: str):
        data: pl.DataFrame = pl.read_excel(path)
        # lower
        if self.lower:
            data = data.with_columns(pl.col("message").str.to_lowercase())
        # drop nulls and duplicates
        data = data.drop_nulls().unique(subset="message")
        # apply filters
        data = data.with_columns(
            pl.col("message").map_elements(self._combined_filter, return_dtype=pl.String)
        )
        return data
