from enum import Enum


class TaskType(Enum):
    CLASSIFICATION = "classification"
    SIGMENTATION = "sigmentation"
    DETECTION = "detection"
