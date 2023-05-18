import torch
import logging
from .torch_model import PcaModel
from anomalib.models.components import AnomalyModule
from typing import Dict, Union, List
from torch import Tensor
logger = logging.getLogger(__name__)

class Pca(AnomalyModule):
    """PCA: Principal Component Analysis.

    Args:
        n_components (float, optional): The proportion of principal components to total components. Defaults to 0.97.
        threshold (float, optional): Defaults to 0.4.
    """
    def __init__(
        self,
        backbone: str,
        layer: str,
        pre_trained: bool = True,
        pooling_kernel_size: int = 4,
        n_components: float = 0.97,
        threshold: float = 0.4
    ):
        super().__init__()
        self.model: PcaModel = PcaModel(
            backbone=backbone,
            layer=layer,
            pre_trained=pre_trained,
            pooling_kernel_size=pooling_kernel_size,
            n_components=n_components,
            threshold = threshold)
        self.embeddings: List[Tensor] = []

    @staticmethod
    def configure_optimizers() -> None:  # pylint: disable=arguments-differ
        """DFM doesn't require optimization, therefore returns no optimizers."""
        return None

    def training_step(self, batch, _):  # pylint: disable=arguments-differ
        """Training Step of DFM.

        For each batch, features are extracted from the CNN.

        Args:
          batch (Dict[str, Tensor]): Input batch
          _: Index of the batch.

        Returns:
          Deep CNN features.
        """
        embedding = self.model.get_features(batch["image"]).squeeze()

        # NOTE: `self.embedding` appends each batch embedding to
        #   store the training set embedding. We manually append these
        #   values mainly due to the new order of hooks introduced after PL v1.4.0
        #   https://github.com/PyTorchLightning/pytorch-lightning/pull/7357
        self.embeddings.append(embedding)

    def on_validation_start(self) -> None:
        """Fit a PCA transformation and a Gaussian model to dataset."""
        # NOTE: Previous anomalib versions fit Gaussian at the end of the epoch.
        #   This is not possible anymore with PyTorch Lightning v1.4.0 since validation
        #   is run within train epoch.
        logger.info("Aggregating the embedding extracted from the training set.")
        embeddings = torch.vstack(self.embeddings)

        logger.info("Fitting a PCA and a Gaussian model to dataset.")
        self.model.fit(embeddings)

    def validation_step(self, batch, _):  # pylint: disable=arguments-differ
        """Validation Step of DFM.

        Similar to the training step, features are extracted from the CNN for each batch.

        Args:
          batch (List[Dict[str, Any]]): Input batch

        Returns:
          Dictionary containing FRE anomaly scores and ground-truth.
        """
        batch["pred_scores"] = self.model(batch["image"])

        return batch


class PcaLightning(Pca):
    """PCA: Principal Component Analysis.

    Args:
        hparams (Union[DictConfig, ListConfig]): Model params
    """

    def __init__(self, hparams) -> None:
        super().__init__(
            backbone=hparams.model.backbone,
            layer=hparams.model.layer,
            pre_trained=hparams.model.pre_trained,
            pooling_kernel_size=hparams.model.pooling_kernel_size,
            n_components=hparams.model.n_components,
            threshold=hparams.model.threshold,
        )
        self.hparams  # type: ignore
        self.save_hyperparameters(hparams)