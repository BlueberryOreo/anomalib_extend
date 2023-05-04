from abc import ABC, abstractmethod
from data.task_type import TaskType

# 数据增强工具
import albumentations as A
from pandas import DataFrame

from torch.utils.data import Dataset


class AnomalousDataset(Dataset, ABC):
    def __init__(self, task: TaskType, transform: A.Compose = None):
        super().__init__()
        self.task = task
        self.transform = transform
        self._samples: DataFrame

    def is_setup(self) -> bool:
        return isinstance(self._samples, DataFrame)

    def get_samples(self) -> DataFrame:
        if not self.is_setup():
            raise RuntimeError("Dataset is not set up yet. Call setup() first.")
        return self._samples

    def setup(self) -> None:
        if not self.is_setup():
            self._setup()
        assert self.is_setup(), "setup() should set self._samples"

    @abstractmethod
    def _setup(self) -> DataFrame:
        raise NotImplementedError
