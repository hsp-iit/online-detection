from abc import ABC, abstractmethod


class ClassifierAbstract(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self, dataset) -> None:
        pass

    @abstractmethod
    def predict(self, dataset) -> None:
        pass

    @abstractmethod
    def test(self, dataset) -> None:
        pass
