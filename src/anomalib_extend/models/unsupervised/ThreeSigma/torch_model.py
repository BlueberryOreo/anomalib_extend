from torch import Tensor
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from anomalib.models.components import DynamicBufferModule, FeatureExtractor

class ThreeSigma_Model(DynamicBufferModule):
    def __init__(self,
        backbone:str,
        layer:str,
        pre_trained: bool = True,
        pooling_kernel_size: int = 4,
    ):
        super().__init__()
        self.scores: List[int]
        self.mu = []
        self.sigma = []
        self.val=0
        self.pooling_kernel_size = pooling_kernel_size
        self.feature_extractor = FeatureExtractor(
            backbone=backbone, pre_trained=pre_trained, layers=[layer]
        ).eval()

    def get_features(self, batch: Tensor) -> Tensor:#多维矩阵映射到二维
        self.feature_extractor.eval()
        features = self.feature_extractor(batch)
        for layer in features:
            batch_size = len(features[layer])
            if self.pooling_kernel_size > 1:
                features[layer] = F.avg_pool2d(input=features[layer], kernel_size=self.pooling_kernel_size)
            features[layer] = features[layer].view(batch_size, -1)

        features = torch.cat(list(features.values())).detach()
        return features



    def three_sigma_gaussian(self, test):  # 按照正态分布计算测试集的密度值
        sig_det=float(np.linalg.det(self.sigma))  # 计算det(Σ)
        sig_inv = np.linalg.inv(self.sigma)  # Σ的逆矩阵
        r = []
        #print(2)
        m, n = np.shape(test)
        for x in test:
            x = np.mat(x).T - self.mu
            temp=-x.T * sig_inv * x / 2
            temp=int(temp)
            temp=self.linear_mapping(temp)
            temp=np.matrix(temp)
            g = np.exp(temp) * ((2 * np.pi) ** (-n / 2) * (sig_det ** (-0.5)))
            r.append(g[0, 0])
        return r

    def three_sigma_caculate(self, train: Tensor):  # 计算训练矩阵中的均值向量和协方差矩阵
        X_train = np.mat(train).T
        self.mu = np.mean(X_train, axis=1)  # 计算均值向量
        self.sigma = np.cov(X_train)  # 计算协方差矩阵
        #print(self.sigma)
        #print(1)
        train_gaussian=self.three_sigma_gaussian(train)
        #print(train_gaussian)
        self.val=np.nanmin(train_gaussian)
        #print(self.val)

    def score(self, features: Tensor) -> Tensor:
        test_gaussian = self.three_sigma_gaussian(features)
        #print(test_gaussian)
        score=[]
        for data in test_gaussian:
            if(np.isnan(data)):
                score.append(-1)#无法进行矩阵运算，属于坏数据
            else:
                score.append(int(data < self.val))
        #print(score)
        return torch.tensor(score)



    def fit(self, dataset: Tensor) -> None:
        X = np.mat(dataset.T)
        self.mu = np.mean(X, axis=1)
        self.sigma = np.mat(np.cov(X))




    def forward(self, batch: Tensor) -> Tensor:
        """Computer score from input images.

        Args:
            batch (Tensor): Input images

        Returns:
            Tensor: Scores
        """
        feature_vector = self.get_features(batch)
        return self.score(feature_vector.view(feature_vector.shape[:2]))


