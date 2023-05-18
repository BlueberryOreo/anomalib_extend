import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
import math
from anomalib_extend.models.components import FeatureExtractor
from sklearn.decomposition import PCA
from typing import Dict

class PcaModel(nn.Module):
    """Model for the PCA algorithm

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
            threshold: float = 0.4,
            score_type: str = "fre",
    ):
        super().__init__()
        self.backbone = backbone
        self.pooling_kernel_size = pooling_kernel_size
        self.n_components = n_components
        self.threshold = threshold
        self.pca_model = PCA(n_components=self.n_components)
        self.score_type = score_type
        self.feature_extractor = FeatureExtractor(
            backbone=self.backbone, pre_trained=pre_trained, layers=[layer]
        ).eval()

    def fit(self, dataset: Tensor) -> None:
        """Fit a pca transformation and a Gaussian model to dataset.

        Args:
            dataset (Tensor): Input dataset to fit the model.
        """

        self.pca_model.fit(dataset)


    def normalization(self, x):
        """normalization

        """
        return math.atan(x-10) / math.pi + 0.5

    def score(self, features: Tensor) -> Tensor:
        """Compute scores.

        Scores are either PCA-based feature reconstruction error (FRE) scores or
        the Gaussian density-based NLL scores

        Args:
            features (torch.Tensor): semantic features on which PCA and density modeling is performed.

        Returns:
            score (Tensor): numpy array of scores
        """
        score = []
        e = self.pca_model.components_  # 特征向量
        lambdas = self.pca_model.singular_values_  # 特征值
        # print("e.shape = ", np.array(e).shape)
        # print("lambdas.shape = ", np.array(lambdas).shape)
        for xi in features:
            # print("xi.shape = ", np.array(xi).shape)
            # transformed_xi = self.pca_model.transform([xi])

            # print("transformtest = ", transformed_test)
            # print("e = ", e)
            # print("lambdas = ", lambdas)
            # print("transformxi.shape = ", np.array(transformed_xi).shape)
            d = 0  # 偏离程度
            # print("len(list(lambdas)) = ", len(list(lambdas)))
            for j in range(len(list(lambdas))):
                # print(np.array(xi))
                # print(np.array(e[j]))
                d += np.dot(np.array(xi).T, np.array(e[j])) ** 2 / lambdas[j]
            abnormal_score = self.normalization(d);
            if (abnormal_score > self.threshold):
                score.append(1)  # 异常
            else:
                score.append(0)  # 正常
            # print("len(self.scores)", len(self.scores))

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

