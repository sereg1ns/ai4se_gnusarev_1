import typing as tp

from torch.utils.data import Dataset
import torch
import polars as pl
from transformers import RobertaTokenizer


class ToxicReviewDataset(Dataset):
    def __init__(
        self,
        data_path: str,
    ) -> Dataset:
        super(ToxicReviewDataset).__init__()
        self.data = pl.read_excel(data_path)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

def collate_with_tokenizer(tokenizer: RobertaTokenizer) -> tp.Callable[[tp.Any], tp.Tuple[tp.Dict[str, torch.Tensor], torch.Tensor]]:
    def collate_fn(batch: tp.List[pl.DataFrame]) -> tp.Tuple[tp.Dict[str, torch.Tensor], torch.Tensor]:
        concat_df = pl.concat(batch)
        labels = concat_df["is_toxic"].to_torch()
        tokens = tokenizer(
            concat_df["message"].to_list(),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return tokens, labels
    return collate_fn

