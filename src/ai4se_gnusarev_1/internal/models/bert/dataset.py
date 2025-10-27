from torch.utils.data import Dataset
import polars as pl


class ToxicReviewDataset(Dataset):
    def __init__(
        self,
        data_path: str,
    ) -> Dataset:
        super(ToxicReviewDataset).__init__()
        self.data = pl.read_exel(data_path)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)
