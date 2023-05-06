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
        self._samples: DataFrame = None

    def is_setup(self) -> bool:
        return isinstance(self._samples, DataFrame)

    def get_samples(self) -> DataFrame:
        if not self.is_setup():
            raise RuntimeError("Dataset is not set up yet. Call setup() first.")
        return self._samples

    def set_samples(self, samples: DataFrame):
        assert isinstance(samples, DataFrame), "samples must be a pandas.DataFrame, found {}".format(type(samples))
        self._samples = samples.sort_values(by="image_path", ignore_index=True)

    def setup(self) -> None:
        if not self.is_setup():
            self._setup()
        assert self.is_setup(), "setup() should set self._samples"

    def __getitem__(self, index: int):

        image_path = self._samples.iloc[index].image_path
        mask_path = self._samples.iloc[index].mask_path
        label_index = self._samples.iloc[index].label_index


    @abstractmethod
    def _setup(self) -> DataFrame:
        raise NotImplementedError
