
from enum import Enum


class ShotType(Enum):
    """Shot type used to decide whether to enlarge the dataset by transform images."""

    FEW_SHOT = "few_shot"
    FULL_SHOT = "full_shot"
    