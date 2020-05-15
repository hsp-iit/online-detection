from abc import ABC, abstractmethod


class RegionRefinerAbstract(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def loadRegionRefiner(self) -> None:
        pass

    @abstractmethod
    def trainRegionRefiner(self, dataset) -> None:
        pass

    @abstractmethod
    def testRegionRefiner(self, dataset) -> None:
        pass

    @abstractmethod
    def predict(self, dataset) -> None:
        pass
