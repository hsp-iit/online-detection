from abc import ABC, abstractmethod


class NegativeSelectorAbstract(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def selectPositives(self, dataset) -> None:
        pass
