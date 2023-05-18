import torch
import logging
import numpy as np
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from torch import Tensor, nn
from typing import Union, Optional
from anomalib_extend.models.components import AnomalyModule
from anomalib_extend.models.AutoEncoder.torch_model import AutoEncoderModel
from anomalib_extend.models.AutoEncoder.loss import AutoEncoderLoss

logger = logging.getLogger(__name__)
class AutoEncoder(AnomalyModule):
    def __init__(self,
                 backbone: str,
                 layer: str,
                 threshold: float,
                 pre_trained: bool = True,
                 pooling_kernel_size: int = 4,
                 ):
        super().__init__()
        self.model = AutoEncoderModel(
            backbone=backbone,
            pre_trained=pre_trained,
            layer=layer,
            pooling_kernel_size=pooling_kernel_size,
            threshold=threshold
        )
        self.loss = AutoEncoderLoss()

    def training_step(self, batch, _):
        embedding = self.model.get_features(batch["image"]).squeeze()
        decoderout = self.model(embedding)
        loss = self.loss(embedding, decoderout)
        return {"loss": loss}

    def on_validation_start(self) -> None:
        logger.info("validating...")

    def validation_step(self, batch, _) -> dict:
        embedding = self.model.get_features(batch["image"]).squeeze()
        prediction = self.model(embedding)
        score = np.mean(np.abs(prediction - embedding), dim=1)
        batch["anomaly_maps"] = score > self.threshold
        return batch

class AutoencoderLightning(AutoEncoder):
    def __init__(self, hparams: Union[DictConfig, ListConfig]):
        super().__init__(
            backbone=hparams.model.backbone,
            layer=hparams.model.layer,
            pre_trained=hparams.model.pre_trained,
            pooling_kernel_size=hparams.model.pooling_kernel_size,
            threshold=hparams.model.threshold

        )
        self.hparams: Union[DictConfig, ListConfig]  # type: ignore
        self.save_hyperparameters(hparams)

    def configure_callbacks(self):
        """Configure model-specific callbacks.

        Note:
            This method is used for the existing CLI.
            When PL CLI is introduced, configure callback method will be
                deprecated, and callbacks will be configured from either
                config.yaml file or from CLI.
        """
        early_stopping = EarlyStopping(
            monitor=self.hparams.model.early_stopping.metric,
            patience=self.hparams.model.early_stopping.patience,
            mode=self.hparams.model.early_stopping.mode,
        )
        return [early_stopping]

    def configure_optimizers(self):  # pylint: disable=arguments-differ
        """Configure the Adam optimizer."""
        return torch.optim.Adam(params=self.model.parameters(), lr=self.hparams.model.lr)


