from abc import ABC, abstractmethod


class FeatureExtractorAbstract(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def loadFeatureExtractor(self) -> None:
        pass

    @abstractmethod
    def trainFeatureExtractor(self, dataset) -> None:
        pass

    @abstractmethod
    def extractFeatures(self, dataset) -> None:
        pass
