from abc import ABC, abstractmethod
from data.task_type import TaskType
from torch.utils.data import Dataset


class AnomalousDataset(Dataset, ABC):
    def __init__(self, task: TaskType):
        super().__init__()
        self.task = task
