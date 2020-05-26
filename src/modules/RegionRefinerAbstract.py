from abc import ABC, abstractmethod


class RegionRefinerAbstract(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def loadRegionRefiner(self) -> None:
        pass

    @abstractmethod
    def trainRegionRefiner(self) -> None:
        pass

    @abstractmethod
    def testRegionRefiner(self) -> None:
        pass

    @abstractmethod
    def predict(self) -> None:
        pass
