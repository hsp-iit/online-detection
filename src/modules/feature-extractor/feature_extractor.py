import os
import errno
import sys

from trainer_feature_task import TrainerFeatureTask
from feature_extractor_detector import FeatureExtractorDetector
from feature_extractor_loader import LoaderFeatureExtractor
basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
from FeatureExtractorAbstract import FeatureExtractorAbstract


class FeatureExtractor(FeatureExtractorAbstract):
    def __init__(self, cfg_path_feature_task=None, cfg_path_target_task=None):
        self.cfg_path_feature_task = cfg_path_feature_task
        self.cfg_path_target_task = cfg_path_target_task

    def loadFeatureExtractor(self):
        loader = LoaderFeatureExtractor(self.cfg_path_target_task)
        try:
            loader()
        except OSError:
            print('Feature extractor not found.')
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT))

    def trainFeatureExtractor(self):
        # call class to train from scratch a model on the feature task
        trainer = TrainerFeatureTask(self.cfg_path_feature_task)
        trainer()

        # maybe this part can be included in the training class
        """
        if len(models) == 0:
            model = models[0]
        else:
            model = self.select_feature_extractor(models)

        return model   #or return path to models if they are saved somewhere and maybe some metrics such as mAP
        """
    """
    def select_feature_extractor(self, models):
        # evaluate models computed by train_model_on_feature_task, maybe using some metrics learning
        model = []

        return model

    def extract_rpn_features(self):
        # call class to extract rpn features:
        features = []

        return features
    """
    def extractFeatures(self):
        # call class to extract detector features:
        feature_extractor = FeatureExtractorDetector(self.cfg_path_target_task)
        features = feature_extractor()

        return features
    """
    def extract_detector_features_from_updated_RPN(self):
        # call class to extract detector features embedding rpn update:
        features = []

        return features
    """
