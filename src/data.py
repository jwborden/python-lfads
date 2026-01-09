"""
Dataset for ephys data
"""

import os
from pathlib import Path

import pandas as pd  # type: ignore
import torch
from torch.utils.data import Dataset

from src.utils import wd


class OneDs(Dataset):
    def __init__(self, train: bool, ins: str = "") -> None:
        """
        :param test: is this the train (true) or test (false) set?
        """
        super().__init__()
        self.wd: Path = wd()
        self.dir = self.wd / "data" / ("train" if train else "test")
        self.files: list[str] = []
        self.spikes_shape = (128, 160)

        if ins == "" or ins is None:
            for entry in os.scandir(self.dir):
                if entry.name.endswith(".parquet"):
                    self.files.append(str(self.dir / entry.name))
        else:
            for entry in os.scandir(self.dir):
                if entry.name.endswith(".parquet") and ins in entry.name:
                    self.files.append(str(self.dir / entry.name))

        return None

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i) -> torch.Tensor:
        fp = self.files[i]
        x: torch.Tensor = torch.from_numpy(pd.read_parquet(fp).values)
        return x
