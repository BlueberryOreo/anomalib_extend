import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from pyod.models.ocsvm import OCSVM
from anomalib.models.components import PCA, DynamicBufferModule, FeatureExtractor


class ocsvmModel(nn.Module):
    """Model for the DFM algorithm.

    Args:
        backbone (str): Pre-trained model backbone.
        layer (str): Layer from which to extract features.
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
        pooling_kernel_size (int, optional): Kernel size to pool features extracted from the CNN.
        n_comps (float, optional): Ratio from which number of components for PCA are calculated. Defaults to 0.97.
        score_type (str, optional): Scoring type. Options are `fre` and `nll`. Defaults to "fre".
    """

    def __init__(
        self,
        backbone: str,
        layer: str,
        pre_trained: bool = True,
        pooling_kernel_size: int = 4,
    ):
        super().__init__()
        self.backbone = backbone
        self.pooling_kernel_size = pooling_kernel_size
        self.model = OCSVM()
        self.feature_extractor = FeatureExtractor(
            backbone=self.backbone, pre_trained=pre_trained, layers=[layer]
        ).eval()

    def fit(self, dataset: Tensor) -> None:
        """Fit a one-class-svm model to dataset.

        Args:
            dataset (Tensor): Input dataset to fit the model.
        """
        self.model.fit(dataset)

    def score(self, features: Tensor) -> Tensor:
        """Compute scores.

        Scores are either PCA-based feature reconstruction error (FRE) scores or
        the Gaussian density-based NLL scores

        Args:
            features (torch.Tensor): semantic features on which PCA and density modeling is performed.

        Returns:
            score (Tensor): numpy array of scores
        """
        score = self.model.decision_function(features)

        return torch.tensor(score)

    def get_features(self, batch: Tensor) -> Tensor:
        """Extract features from the pretrained network.

        Args:
            batch (Tensor): Image batch.

        Returns:
            Tensor: Tensor containing extracted features.
        """
        self.feature_extractor.eval()
        features = self.feature_extractor(batch)
        for layer in features:
            batch_size = len(features[layer])
            if self.pooling_kernel_size > 1:
                features[layer] = F.avg_pool2d(input=features[layer], kernel_size=self.pooling_kernel_size)
            features[layer] = features[layer].view(batch_size, -1)

        features = torch.cat(list(features.values())).detach()
        return features

    def forward(self, batch: Tensor) -> Tensor:
        """Computer score from input images.

        Args:
            batch (Tensor): Input images

        Returns:
            Tensor: Scores
        """
        feature_vector = self.get_features(batch)
        return self.score(feature_vector.view(feature_vector.shape[:2]))
