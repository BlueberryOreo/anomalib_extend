"""Load Anomaly Model."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from importlib import import_module
from typing import List, Union

from omegaconf import DictConfig, ListConfig
from torch import load

from .components import AnomalyModule
from .unsupervised import (
    AutoEncoder,
    Cflow,
    Cof,
    Dfkde,
    Dfm,
    Draem,
    Fastflow,
    Ganomaly,
    Lof,
    OneClassSvm,
    Padim,
    Patchcore,
    Pca,
    ReverseDistillation,
    Sos,
    Stfpm,
    ThreeSigma,
)

__all__ = [
    "AutoEncoder",
    "Cflow",
    "Cof",
    "Dfkde",
    "Dfm",
    "Draem",
    "Fastflow",
    "Ganomaly",
    "Lof",
    "OneClassSvm",
    "Padim",
    "Patchcore",
    "Pca",
    "ReverseDistillation",
    "Sos",
    "Stfpm",
    "ThreeSigma",
]

logger = logging.getLogger(__name__)


def _snake_to_pascal_case(model_name: str) -> str:
    """Convert model name from snake case to Pascal case.

    Args:
        model_name (str): Model name in snake case.

    Returns:
        str: Model name in Pascal case.
    """
    return "".join([split.capitalize() for split in model_name.split("_")])


def get_model(config: Union[DictConfig, ListConfig]) -> AnomalyModule:
    """Load model from the configuration file.

    Works only when the convention for model naming is followed.

    The convention for writing model classes is
    `anomalib.models.<model_name>.lightning_model.<ModelName>Lightning`
    `anomalib.models.stfpm.lightning_model.StfpmLightning`

    Args:
        config (Union[DictConfig, ListConfig]): Config.yaml loaded using OmegaConf

    Raises:
        ValueError: If unsupported model is passed

    Returns:
        AnomalyModule: Anomaly Model
    """
    logger.info("Loading the model.")

    model_list: List[str] = [
        "AutoEncoder",
        "cflow",
        "cof",
        "dfkde",
        "dfm",
        "draem",
        "fastflow",
        "ganomaly",
        "lof",
        "OneClassSvm",
        "padim",
        "patchcore",
        "pca",
        "reverse_distillation",
        "sos",
        "stfpm",
        "ThreeSigma",
    ]
    model: AnomalyModule

    if config.model.name in model_list:
        module = import_module(f"anomalib_extend.models.unsupervised.{config.model.name}")
        model = getattr(module, f"{_snake_to_pascal_case(config.model.name)}Lightning")(config)

    else:
        raise ValueError(f"Unknown model {config.model.name}!")

    if "init_weights" in config.keys() and config.init_weights:
        model.load_state_dict(load(os.path.join(config.project.path, config.init_weights))["state_dict"], strict=False)

    return model
