from abc import ABC, abstractmethod


class FeatureExtractorAbstract(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def extractRPNFeatures(self) -> None:
        pass

    @abstractmethod
    def extractFeatures(self) -> None:
        pass

    @abstractmethod
    def trainFeatureExtractor(self) -> None:
        pass
