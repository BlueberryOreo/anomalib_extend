import logging
from typing import List, Union

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from torch import Tensor

from anomalib.models.components import AnomalyModule

from .torch_model import ThreeSigma_Model

logger = logging.getLogger(__name__)


@MODEL_REGISTRY
class ThreeSigma(AnomalyModule):
    """ThreeSigma.

    Args:
        backbone (str): Backbone CNN network
        layer (str): Layer to extract features from the backbone CNN
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
        pooling_kernel_size (int, optional): Kernel size to pool features extracted from the CNN.
            Defaults to 4.
    """

    def __init__(
        self,
        backbone: str,
        layer: str,
        pre_trained: bool = True,
        pooling_kernel_size: int = 4,
    ):
        super().__init__()

        self.model: ThreeSigma_Model = ThreeSigma_Model(
            backbone=backbone,
            pre_trained=pre_trained,
            layer=layer,
            pooling_kernel_size=pooling_kernel_size,
        )
        self.embeddings: List[Tensor] = []

    @staticmethod
    def configure_optimizers() -> None:  # pylint: disable=arguments-differ
        """ThreeSigma doesn't require optimization, therefore returns no optimizers."""
        return None

    def training_step(self, batch, _):  # pylint: disable=arguments-differ
        """Training Step of ThreeSigma.

        For each batch, features are extracted from the CNN.

        Args:
          batch (Dict[str, Tensor]): Input batch
          _: Index of the batch.

        Returns:
          Deep CNN features.
        """
        embedding = self.model.get_features(batch["image"]).squeeze()
        self.embeddings.append(embedding)

    def on_validation_start(self) -> None:
        """Fit a PCA transformation and a Gaussian model to dataset."""
        # NOTE: Previous anomalib versions fit Gaussian at the end of the epoch.
        #   This is not possible anymore with PyTorch Lightning v1.4.0 since validation
        #   is run within train epoch.
        logger.info("Aggregating the embedding extracted from the training set.")
        embeddings = torch.vstack(self.embeddings)
        tmp = embeddings.cpu().numpy()
        logger.info("Fitting a Threesigma model to dataset.")
        self.model.fit(embeddings)

    def validation_step(self, batch, _):  # pylint: disable=arguments-differ
        """Validation Step of ThreeSigma.

        Similar to the training step, features are extracted from the CNN for each batch.

        Args:
          batch (List[Dict[str, Any]]): Input batch

        Returns:
          Dictionary containing FRE anomaly scores and ground-truth.
        """
        batch["pred_scores"] = self.model(batch["image"])

        return batch


class ThreesigmaLightning(ThreeSigma):
    """ThreeSigma.

    Args:
        hparams (Union[DictConfig, ListConfig]): Model params
    """

    def __init__(self, hparams: Union[DictConfig, ListConfig]) -> None:
        super().__init__(
            backbone=hparams.model.backbone,
            layer=hparams.model.layer,
            pre_trained=hparams.model.pre_trained,
            pooling_kernel_size=hparams.model.pooling_kernel_size,
        )
        self.hparams: Union[DictConfig, ListConfig]  # type: ignore
        self.save_hyperparameters(hparams)