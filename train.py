"""Anomalib Training Script.

This script reads the name of the model or config file from command
line, train/test the anomaly model to get quantitative and qualitative
results.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser, Namespace
import logging
import warnings

from pytorch_lightning import Trainer, seed_everything

from anomalib_extend.config import get_configurable_parameters
from anomalib_extend.data import get_datamodule
from anomalib_extend.data.utils import TestSplitMode
from anomalib_extend.models import get_model
from anomalib_extend.utils.callbacks import LoadModelCallback, get_callbacks
from anomalib_extend.utils.loggers import configure_logger, get_experiment_logger

logger = logging.getLogger("anomalib_extend")


def get_args() -> Namespace:
    """Get command line arguments.

    Returns:
        Namespace: List of arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="patchcore", help="Name of the algorithm to train/test")
    parser.add_argument("--config", type=str, 
                        default="D:\\learn\\courseware\\project\\异常检测\\第三步\\anomalib-main\\src\\anomalib\\models\\patchcore\\config.yaml",
                          help="Path to a model config file")
    parser.add_argument("--log-level", type=str, default="INFO", help="<DEBUG, INFO, WARNING, ERROR>")

    args = parser.parse_args()
    return args


def train():
    """Train an anomaly classification or segmentation model based on a provided configuration file."""
    args = get_args()
    if args.log_level == "ERROR":
        warnings.filterwarnings("ignore")

    config = get_configurable_parameters(model_name=args.model, config_path=args.config)
    if config.project.get("seed") is not None:
        seed_everything(config.project.seed)
    
    datamodule = get_datamodule(config)
    model = get_model(config)
    experiment_logger = get_experiment_logger(config)
    callbacks = get_callbacks(config)

    trainer = Trainer(**config.trainer, logger=experiment_logger, callbacks=callbacks)
    logger.info("Training the model.")
    trainer.fit(model=model, datamodule=datamodule)

    logger.info("Loading the best model weights.")
    load_model_callback = LoadModelCallback(weights_path=trainer.checkpoint_callback.best_model_path)
    trainer.callbacks.insert(0, load_model_callback)

    if config.dataset.test_split_mode == TestSplitMode.NONE:
        logger.info("No test set provided. Skipping test stage.")
    else:
        logger.info("Testing the model.")
        trainer.test(model=model, datamodule=datamodule)


if __name__ == '__main__':
    train()
