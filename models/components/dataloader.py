from abc import ABC, abstractmethod


class DataModule(ABC):
    def __init__(
            self,
            train_batch_size: int,
            eval_batch_size: int,

    ):
        pass
