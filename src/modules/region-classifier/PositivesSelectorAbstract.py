from abc import ABC, abstractmethod


class PositivesSelectorAbstract(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def selectPositives(self, dataset) -> None:
        pass

    @abstractmethod
    def get_num_classes(self) -> None:
        pass
