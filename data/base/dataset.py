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

