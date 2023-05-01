from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class AnomalousDataset(Dataset, ABC):
    def __init__(self, task):
        super().__init__()
