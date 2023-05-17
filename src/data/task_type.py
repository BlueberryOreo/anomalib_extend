from enum import Enum


class TaskType(Enum):
    CLASSIFICATION = "classification"
    SIGMENTATION = "sigmentation"
    DETECTION = "detection"


class DataType(Enum):
    FEW_SHOT = "few_shot"
    FULL_SHOT = "full_shot"
