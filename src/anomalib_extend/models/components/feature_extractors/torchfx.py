"""Feature Extractor based on TorchFX."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import torch
from torch import Tensor, nn
from torch.fx.graph_module import GraphModule
from torchvision.models._api import WeightsEnum
from torchvision.models.feature_extraction import create_feature_extractor


@dataclass
class BackboneParams:
    """Used for serializing the backbone."""

    class_path: Union[str, nn.Module]
    init_args: Dict = field(default_factory=dict)


class TorchFXFeatureExtractor(nn.Module):
    """Extract features from a CNN.

    Args:
        backbone (Union[str, BackboneParams, Dict, nn.Module]): The backbone to which the feature extraction hooks are
            attached. If the name is provided, the model is loaded from torchvision. Otherwise, the model class can be
            provided and it will try to load the weights from the provided weights file.
        return_nodes (Iterable[str]): List of layer names of the backbone to which the hooks are attached.
            You can find the names of these nodes by using ``get_graph_node_names`` function.
        weights (Optional[Union[WeightsEnum,str]]): Weights enum to use for the model. Torchvision models require
            ``WeightsEnum``. These enums are defined in ``torchvision.models.<model>``. You can pass the weights
            path for custom models.
        requires_grad (bool): Models like ``stfpm`` use the feature extractor for training. In such cases we should
            set ``requires_grad`` to ``True``. Default is ``False``.

    Example:
        With torchvision models:

            >>> import torch
            >>> from anomalib.models.components.feature_extractors import TorchFXFeatureExtractor
            >>> from torchvision.models.efficientnet import EfficientNet_B5_Weights
            >>> feature_extractor = TorchFXFeatureExtractor(
                    backbone="efficientnet_b5",
                    return_nodes=["features.6.8"],
                    weights=EfficientNet_B5_Weights.DEFAULT
                )
            >>> input = torch.rand((32, 3, 256, 256))
            >>> features = feature_extractor(input)
            >>> [layer for layer in features.keys()]
                ["features.6.8"]
            >>> [feature.shape for feature in features.values()]
                [torch.Size([32, 304, 8, 8])]

        With custom models:

            >>> from anomalib.models.components.feature_extractors import TorchFXFeatureExtractor
            >>> feature_extractor = TorchFXFeatureExtractor(
                    "path.to.CustomModel", ["linear_relu_stack.3"], weights="path/to/weights.pth"
                )
            >>> input = torch.randn(1, 1, 28, 28)
            >>> features = feature_extractor(input)
            >>> [layer for layer in features.keys()]
                ["linear_relu_stack.3"]
    """

    def __init__(
        self,
        backbone: Union[str, BackboneParams, Dict, nn.Module],
        return_nodes: List[str],
        weights: Optional[Union[WeightsEnum, str]] = None,
        requires_grad: bool = False,
    ):
        super().__init__()
        if isinstance(backbone, dict):
            backbone = BackboneParams(**backbone)
        elif not isinstance(backbone, BackboneParams):  # if str or nn.Module
            backbone = BackboneParams(class_path=backbone)

        self.feature_extractor = self.initialize_feature_extractor(backbone, return_nodes, weights, requires_grad)

    def initialize_feature_extractor(
        self,
        backbone: BackboneParams,
        return_nodes: List[str],
        weights: Optional[Union[WeightsEnum, str]] = None,
        requires_grad: bool = False,
    ) -> Union[GraphModule, nn.Module]:
        """Extract features from a CNN.

        Args:
            backbone (Union[str, BackboneParams]): The backbone to which the feature extraction hooks are attached.
                If the name is provided, the model is loaded from torchvision. Otherwise, the model class can be
                provided and it will try to load the weights from the provided weights file.
            return_nodes (Iterable[str]): List of layer names of the backbone to which the hooks are attached.
                You can find the names of these nodes by using ``get_graph_node_names`` function.
            weights (Optional[Union[WeightsEnum,str]]): Weights enum to use for the model. Torchvision models require
                ``WeightsEnum``. These enums are defined in ``torchvision.models.<model>``. You can pass the weights
                path for custom models.
            requires_grad (bool): Models like ``stfpm`` use the feature extractor for training. In such cases we should
                set ``requires_grad`` to ``True``. Default is ``False``.

        Returns:
            Feature Extractor based on TorchFX.
        """
        if isinstance(backbone.class_path, str):
            backbone_class = self._get_backbone_class(backbone.class_path)
            backbone_model = backbone_class(weights=weights, **backbone.init_args)
        else:
            backbone_class = backbone.class_path
            backbone_model = backbone_class(**backbone.init_args)
        if isinstance(weights, WeightsEnum):  # torchvision models
            feature_extractor = create_feature_extractor(model=backbone_model, return_nodes=return_nodes)
        else:
            if weights is not None:
                assert isinstance(weights, str), "Weights should point to a path"
                model_weights = torch.load(weights)
                if "state_dict" in model_weights:
                    model_weights = model_weights["state_dict"]
                backbone_model.load_state_dict(model_weights)
            feature_extractor = create_feature_extractor(backbone_model, return_nodes)

        if not requires_grad:
            feature_extractor.eval()
            for param in feature_extractor.parameters():
                param.requires_grad_(False)

        return feature_extractor

    @staticmethod
    def _get_backbone_class(backbone: str) -> Callable[..., nn.Module]:
        """Get the backbone class from the provided path.

        If only the model name is provided, it will try to load the model from torchvision.

        Example:
            >>> from anomalib.models.components.feature_extractors import TorchFXFeatureExtractor
            >>> TorchFXFeatureExtractor._get_backbone_class("efficientnet_b5")
            <function torchvision.models.efficientnet.efficientnet_b5(
                *,
                weights: Union[torchvision.models.efficientnet.EfficientNet_B5_Weights, NoneType] = None,
                progress: bool = True,
                **kwargs: Any
                ) -> torchvision.models.efficientnet.EfficientNet>

            >>> TorchFXFeatureExtractor._get_backbone_class("path.to.CustomModel")
            <class 'path.to.CustomModel'>

        Args:
            backbone (str): Path to the backbone class.

        Returns:
            Backbone class.
        """
        try:
            if len(backbone.split(".")) > 1:
                # assumes that the entire class path is provided
                models = importlib.import_module(".".join(backbone.split(".")[:-1]))
                backbone_class = getattr(models, backbone.split(".")[-1])
            else:
                models = importlib.import_module("torchvision.models")
                backbone_class = getattr(models, backbone)
        except ModuleNotFoundError as exception:
            raise ModuleNotFoundError(
                f"Backbone {backbone} not found in torchvision.models nor in {backbone} module."
            ) from exception

        return backbone_class

    def forward(self, inputs: Tensor) -> Dict[str, Tensor]:
        """Extract features from the input."""
        return self.feature_extractor(inputs)
