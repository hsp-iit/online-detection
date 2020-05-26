from abc import ABC, abstractmethod
import yaml
import os
import sys
basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..')))
from utils import getFeatPath


class RegionClassifierAbstract(ABC):
    def __init__(self, classifier, negative_selector, cfg_path=None):
        if cfg_path is not None:
            self.cfg = yaml.load(open(cfg_path), Loader=yaml.FullLoader)
            self.experiment_name = self.cfg['EXPERIMENT_NAME']
            self.num_classes = self.cfg['NUM_CLASSES']
            self.train_imset = self.cfg['DATASET']['TARGET_TASK']['TRAIN_IMSET']
            self.test_imset = self.cfg['DATASET']['TARGET_TASK']['TEST_IMSET']
            self.classifier_options = self.cfg['ONLINE_REGION_CLASSIFIER']['CLASSIFIER']
            self.feature_folder = getFeatPath(self.cfg)

        else:
            print('Config file path not given. cfg variable set to None.')
            self.cfg = None

        self.classifier = classifier
        self.negative_selector = negative_selector
        # self.experiment_name = experiment_name

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
