"""Augmenter module to generates out-of-distribution samples for the DRAEM implementation."""

# Original Code
# Copyright (c) 2021 VitjanZ
# https://github.com/VitjanZ/DRAEM.
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import glob
import math
import random
from typing import Optional, Tuple, Union

import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
from torch import Tensor
from torchvision.datasets.folder import IMG_EXTENSIONS

