from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from typing import Optional

from pytorch_lightning import LightningDataModule

from data.utils import (
    TestSplitMode,
    ValSplitMode,
    random_split,
    split_by_label
)


class DataModule(LightningDataModule, ABC):
    def __init__(
            self,
            train_batch_size: int,
            eval_batch_size: int,
            num_workers: int,
            val_split_mode: ValSplitMode,
            val_split_ratio: float,
            test_split_mode: Optional[TestSplitMode] = None,
            test_split_ratio: Optional[float] = None,
            seed: Optional[int] = None
    ):
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        
