from abc import ABC, abstractmethod
import yaml


class RegionClassifierAbstract(ABC):
    def __init__(self, classifier, negative_selector, cfg_path=None):
        if cfg_path is not None:
            self.cfg = yaml.load(open(cfg_path), Loader=yaml.FullLoader)
            self.experiment_name = self.cfg['EXPERIMENT_NAME']
            self.num_classes = self.cfg['NUM_CLASSES']
            self.train_imset = self.cfg['DATASET']['TARGET_TASK']['TRAIN_IMSET']
            self.test_imset = self.cfg['DATASET']['TARGET_TASK']['TEST_IMSET']
            self.classifier_options = self.cfg['ONLINE_REGION_CLASSIFIER']['CLASSIFIER']
            self.feature_folder = '' + self.cfg['FEATURE_INFO']['BACKBONE'] + '_ep' \
                                  + str(self.cfg['FEATURE_INFO']['NUM_EPOCHS']) + '_FT' \
                                  + self.cfg['FEATURE_INFO']['FEAT_TASK_NAME'] + '_TT' \
                                  + self.cfg['FEATURE_INFO']['TARGET_TASK_NAME'] + ''

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
