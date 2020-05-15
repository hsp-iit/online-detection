from abc import ABC, abstractmethod


class RegionClassifierAbstract(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def loadRegionClassifier(self) -> None:
        pass

    @abstractmethod
    def trainRegionClassifier(self, dataset) -> None:
        pass

    @abstractmethod
    def testRegionClassifier(self, dataset) -> None:
        pass

    @abstractmethod
    def predict(self, dataset) -> None:
        pass
