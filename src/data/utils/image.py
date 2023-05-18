
import numpy as np
import cv2
from typing import Union, Tuple
from pathlib import Path


def get_image_height_and_width(image_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(image_size, int):
        return (image_size, image_size)
    elif isinstance(image_size, tuple):
        return image_size
    else:
        raise ValueError("``image_size`` could be either int or Tuple[int, int]")


def read_image(path: Union[str, Path], image_size = None) -> np.ndarray:
    path = path if isinstance(path, str) else str(path)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if image_size:
        height, width = get_image_height_and_width(image_size)
        image = cv2.resize(height, width)
    
    return image
