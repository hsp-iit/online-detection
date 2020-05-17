from abc import ABC, abstractmethod


class NegativeSelectorAbstract(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def selectNegatives(self, dataset) -> None:
        pass
