from torch import nn
from anomalib.models.components import FeatureExtractor
from pyod.models.sos import SOS
import torch
import torch.nn.functional as F
import numpy as np


class SosModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        layer: str,
        pre_trained: bool = True,
        pooling_kernel_size: int = 4,
        n_comps: float = 0.97,
        score_type: str = "fre",
    ):
        super().__init__()
        self.backbone = backbone
        self.pooling_kernel_size = pooling_kernel_size
        self.n_components = n_comps
        self.model = SOS()
        self.score_type = score_type
        self.feature_extractor = FeatureExtractor(
            backbone=self.backbone, pre_trained=pre_trained, layers=[layer]
        ).eval()

    def fit(self, dataset):
        """Fit a pca transformation and a Gaussian model to dataset.

        Args:
            dataset (Tensor): Input dataset to fit the model.
        """

        self.model.fit(dataset)


    def score(self, features):
        """Compute scores.

        Returns:
            score (Tensor): numpy array of scores
        """
        score = np.array(self.model.decision_function(features))

        return torch.tensor(score)

    def get_features(self, batch):
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

    def forward(self, batch):
        """Computer score from input images.

        Args:
            batch (Tensor): Input images

        Returns:
            Tensor: Scores
        """
        feature_vector = self.get_features(batch)
        return self.score(feature_vector.view(feature_vector.shape[:2]))