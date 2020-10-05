from abc import ABC, abstractmethod


class AccuracyEvaluatorAbstract(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def evaluateAccuracyDetection(self) -> None:
        pass
