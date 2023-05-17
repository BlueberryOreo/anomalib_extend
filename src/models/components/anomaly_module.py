from abc import ABC, abstractmethod
from torch import nn, Tensor
import pytorch_lightning as pl


class AnomalyModule(ABC, pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.save_hyperparameters()
        self.model: nn.Module
        self.loss: Tensor

    def forward(self, batch) -> Tensor:
        return self.model(batch)

    @abstractmethod
    def configure_optimizers(self) -> None:
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx) -> None:
        pass

    @abstractmethod
    def on_validation_start(self) -> None:
        pass

    @abstractmethod
    def validation_step(self, batch, _) -> dict:
        pass
