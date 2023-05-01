from abc import ABC, abstractmethod
from torch.utils.data import DataLoader


class DataModule(ABC):
    def __init__(
            self,
            train_batch_size: int,
            eval_batch_size: int,

    ):
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        pass
