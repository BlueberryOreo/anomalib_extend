
import logging

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from typing import Optional

from pytorch_lightning import LightningDataModule
from pandas import DataFrame

from data.utils import (
    TestSplitMode,
    ValSplitMode,
    random_split,
    split_by_label
)
from dataset import AnomalibDataset

logger = logging.getLogger(__name__)


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
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.val_split_mode = val_split_mode
        self.val_split_ratio = val_split_ratio
        self.test_split_mode = test_split_mode
        self.test_split_ratio = test_split_ratio
        self.seed = seed

        self.train_data: Optional[AnomalibDataset] = None
        self.val_data: Optional[AnomalibDataset] = None
        self.test_data: Optional[AnomalibDataset] = None

        self._samples: Optional[DataFrame] = None
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup train, validation and test data.

        Args:
          stage: Optional[str]:  Train/Val/Test stages. (Default value = None)
        """
        if not self.is_setup:
            self._setup(stage)
    
    def _setup(self, _stage: Optional[str] = None) -> None:
        """Set up the datasets and perform dynamic subset splitting.

        This method may be overridden in subclass for custom splitting behaviour.

        Note: The stage argument is not used here. This is because, for a given instance of an AnomalibDataModule
        subclass, all three subsets are created at the first call of setup(). This is to accommodate the subset
        splitting behaviour of anomaly tasks, where the validation set is usually extracted from the test set, and
        the test set must therefore be created as early as the `fit` stage.
        """
        assert self.train_data is not None
        assert self.test_data is not None

        self.train_data.setup()
        self.test_data.setup()

    def _create_test_split(self):
        """Obtain the test set based on the settings in the config."""
        if self.test_data.has_normal:
            # split the test data into normal and anomalous so these can be processed separately
            normal_test_data, self.test_data = split_by_label(self.test_data)
        elif self.test_split_mode != TestSplitMode.NONE:
            # when the user did not provide any normal images for testing, we sample some from the training set,
            # except when the user explicitly requested no test splitting.
            logger.info(
                "No normal test image found. Sampling from training set using a split ratio of %d",
                self.test_split_ratio
            )
            self.train_data, normal_test_data = random_split(self.train_data, self.test_split_ratio)
        
        if self.test_split_mode == TestSplitMode.FROM_DIR:
            self.test_data += normal_test_data
        elif self.test_split_mode == TestSplitMode.SYNTHETIC:
            self.test_data = Synth
    
    @property
    def is_setup(self) -> bool:
        """Check if setup() has been called."""
        # at least one of [train_data, val_data, test_data] should be setup
        if self.train_data is not None and self.train_data.is_setup():
            return True
        elif self.val_data is not None and self.val_data.is_setup():
            return True
        elif self.test_data is not None and self.test_data.is_setup():
            return True
        return False
    
