from abc import ABC, abstractmethod
import copy

from torch.utils.data.dataset import ConcatDataset, Dataset
from data.task_type import TaskType
import numpy as np
import pandas as pd
import cv2
from typing import Sequence

# 数据增强工具
import albumentations as A
from pandas import DataFrame

from torch.utils.data import Dataset

from data.utils import read_image
from data.utils import masks_to_boxes


class AnomalousDataset(Dataset, ABC):
    def __init__(self, task: TaskType, transform: A.Compose = None):
        super().__init__()
        self.task = task
        self.transform = transform
        self._samples: DataFrame = None
    
    def __len__(self) -> int:
        """Get length of the dataset"""
        return len(self.samples)
    
    def subsample(self, indices: Sequence[int], inplace: bool = False) -> AnomalousDataset:
        """Subsamples the dataset at the provided indices.

        Args:
            indices (Sequence[int]): Indices at which the dataset is to be subsampled.
            inplace (bool): When true, the subsampling will be performed on the instance itself.
        """
        assert len(set(indices)) == len(indices), "No duplicates allowed in indices."
        dataset = self if inplace else copy.deepcopy(self)
        dataset.samples = self.samples.iloc[indices].reset_index(drop=True)
        return dataset

    @property
    def is_setup(self) -> bool:
        return isinstance(self._samples, DataFrame)

    @property
    def samples(self) -> DataFrame:
        """Get the samples dataframe."""
        if not self.is_setup():
            raise RuntimeError("Dataset is not set up yet. Call setup() first.")
        return self._samples

    @samples.setter
    def samples(self, samples: DataFrame):
        """Overwrite the samples with a new dataframe.

        Args:
            samples (DataFrame): DataFrame with new samples.
        """
        assert isinstance(samples, DataFrame), "samples must be a pandas.DataFrame, found {}".format(type(samples))
        self._samples = samples.sort_values(by="image_path", ignore_index=True)

    @property
    def has_normal(self) -> bool:
        """Check if the dataset contains any normal samples."""
        return 0 in list(self.samples.label_index)
    
    @property
    def has_anomalous(self) -> bool:
        """Check if the dataset contains any anomalous samples."""
        return 1 in list(self.samples.label_index)

    def setup(self) -> None:
        if not self.is_setup():
            self._setup()
        assert self.is_setup(), "setup() should set self._samples"

    def __getitem__(self, index: int):
        """Get dataset item for the index ``index``.

        Args:
            index (int): Index to get the item.

        Returns:
            Union[Dict[str, Tensor], Dict[str, Union[str, Tensor]]]: Dict of image tensor during training.
                Otherwise, Dict containing image path, target path, image tensor, label and transformed bounding box.
        """
        image_path = self._samples.iloc[index].image_path
        mask_path = self._samples.iloc[index].mask_path
        label_index = self._samples.iloc[index].label_index

        image = read_image(image_path)
        item = dict(image_path=image_path, label=label_index)

        if self.task == TaskType.CLASSIFICATION:
            transformed = self.transform(image=image)
            item["image"] = transformed["image"]
        elif self.task in (TaskType.DETECTION, TaskType.SIGMENTATION):
            if label_index == 0:
                mask = np.zeros(shape=image.shape[:2])
            else:
                mask = cv2.imread(mask_path, flags=0) / 255.0
            
            transformed = self.transform(image=image, mask=mask)

            item["image"] = transformed["image"]
            item["mask_path"] = mask_path
            item["mask"] = transformed["mask"]

            if self.task == TaskType.DETECTION:
                # create boxes from masks for detection task
                boxes, _ = masks_to_boxes(item["mask"])
                item["boxes"] = boxes[0]
        else:
            raise ValueError("Unknown task type: {}".format(self.task))
        
        return item
    
    def __add__(self, other: AnomalousDataset) -> AnomalousDataset:
        assert isinstance(other, self.__class__), "Cannot concatenate datasets that are not of the same type."
        assert self.is_setup, "Cannot concatenate uninitialized datasets. Call setup first."
        assert other.is_setup, "Cannot concatenate uninitialized datasets. Call setup first."
        dataset = copy.deepcopy(self)
        dataset.samples = pd.concat([self.samples, other.samples], ignore_index=True)
        return dataset

    @abstractmethod
    def _setup(self) -> DataFrame:
        """Set up the data module.

        This method should return a dataframe that contains the information needed by the dataloader to load each of
        the dataset items into memory.
        The dataframe must at least contain the following columns:
            split: the subset to which the dataset item is assigned.
            image_path: path to file system location where the image is stored.
            label_index: index of the anomaly label, typically 0 for "normal" and 1 for "anomalous".
            mask_path (if task == "segmentation"): path to the ground truth masks (for the anomalous images only).

        Example:
        |---|-------------------|-----------|-------------|------------------|-------|
        |   | image_path        | label     | label_index | mask_path        | split |
        |---|-------------------|-----------|-------------|------------------|-------|
        | 0 | path/to/image.png | anomalous | 1           | path/to/mask.png | train |
        |---|-------------------|-----------|-------------|------------------|-------|
        """
        raise NotImplementedError
