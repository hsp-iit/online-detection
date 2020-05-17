from abc import ABC, abstractmethod


class RegionClassifierAbstract(ABC):
    def __init__(self, experiment_name, classifier, negative_selector):
        self.classifier = classifier
        self.negative_selector = negative_selector
        self.experiment_name = experiment_name

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
