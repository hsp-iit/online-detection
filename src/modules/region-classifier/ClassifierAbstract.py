from abc import ABC, abstractmethod


class RegionClassifierAbstract(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def trainClassifier(self, dataset) -> None:
        pass

    @abstractmethod
    def predict(self, dataset) -> None:
        pass

    @abstractmethod
    def test(self, dataset) -> None:
        pass
