import os
import errno
import sys

from feature_extractor_detector import FeatureExtractorDetector
from feature_extractor_RPN import FeatureExtractorRPN
basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
from FeatureExtractorAbstract import FeatureExtractorAbstract


class FeatureExtractor(FeatureExtractorAbstract):
    def __init__(self, cfg_path_target_task=None, cfg_path_RPN=None):
        self.cfg_path_target_task = cfg_path_target_task
        self.cfg_path_RPN = cfg_path_RPN
        self.falkon_rpn_models = None
        self.regressors_rpn_models = None
        self.stats_rpn = None
        self.regions_post_nms = None

    def extractRPNFeatures(self, is_train, output_dir=None):
        # call class to extract rpn features:
        feature_extractor = FeatureExtractorRPN(self.cfg_path_RPN)
        features = feature_extractor(is_train, output_dir=output_dir)

        return features


    def extractFeatures(self, is_train, output_dir=None):
        # call class to extract detector features:
        feature_extractor = FeatureExtractorDetector(self.cfg_path_target_task)
        feature_extractor.falkon_rpn_models = self.falkon_rpn_models
        feature_extractor.regressors_rpn_models = self.regressors_rpn_models
        feature_extractor.stats_rpn = self.stats_rpn
        if self.regions_post_nms is not None:
            feature_extractor.cfg.MODEL.RPN.POST_NMS_TOP_N_TEST = self.regions_post_nms
        features = feature_extractor(is_train, output_dir=output_dir)

        return features
