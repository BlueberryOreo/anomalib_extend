from enum import Enum


class TaskType(Enum):
    CLASSIFICATION = "classification"
    SIGMENTATION = "sigmentation"
    DETECTION = "detection"
    FEW_SHOT = "few_shot"
    FULL_SHOT = "full_shot"
