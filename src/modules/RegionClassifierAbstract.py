from abc import ABC, abstractmethod
import os
import sys
basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..')))
import yaml


class RegionClassifierAbstract(ABC):
    def __init__(self, classifier, positive_selector, negative_selector, cfg_path=None):
        if cfg_path is not None:
            self.cfg = yaml.load(open(cfg_path), Loader=yaml.FullLoader)
            self.classifier_options = self.cfg['ONLINE_REGION_CLASSIFIER']['CLASSIFIER']
            self.mean = 0
            self.std = 0
            self.mean_norm = 0
            self.is_rpn = False
            self.lam = None
            self.sigma = None

        else:
            print('Config file path not given. cfg variable set to None.')
            self.cfg = None

        self.classifier = classifier

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
