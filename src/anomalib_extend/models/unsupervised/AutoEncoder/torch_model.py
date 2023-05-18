from torch import nn, Tensor
import torch
import torch.nn.functional as F
from anomalib.models.components import FeatureExtractor

class AutoEncoderModel(nn.Module):
    def __init__(self,
                 backbone: str,
                 layer: str,
                 threshold: float,
                 input_size = 256,
                 hidenlayer_size = 128,
                 pre_trained: bool = True,
                 pooling_kernel_size: int = 4,
    ):
        super().__init__()
        self.backbone = backbone
        self.pooling_kernel_size = pooling_kernel_size
        self.threshold = threshold
        self.encoderLayer = nn.Linear(input_size, hidenlayer_size)
        self.decoderLayer = nn.Linear(hidenlayer_size, input_size)
        self.relu = nn.ReLU()
        self.sigmod = nn.Sigmoid()
        self.feature_extractor = FeatureExtractor(
            backbone=self.backbone, pre_trained=pre_trained, layers=[layer]
        ).eval()

    def forward(self, input):
        encoder_linear = self.encoderLayer(input)
        encoder_out = self.relu(encoder_linear)
        decoder_linear = self.decoderLayer(encoder_out)
        res = self.sigmod(decoder_linear)
        return res

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


