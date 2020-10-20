import os
import errno
import sys

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
from FeatureExtractorAbstract import FeatureExtractorAbstract


class FeatureExtractor(FeatureExtractorAbstract):
    def __init__(self, cfg_path_target_task=None, cfg_path_RPN=None, cfg_path_feature_task=None, train_in_cpu=False):
        self.cfg_path_target_task = cfg_path_target_task
        self.cfg_path_RPN = cfg_path_RPN
        self.cfg_path_feature_task = cfg_path_feature_task
        self.falkon_rpn_models = None
        self.regressors_rpn_models = None
        self.stats_rpn = None
        self.falkon_detector_models = None
        self.regressors_detector_models = None
        self.stats_detector = None
        self.regions_post_nms = None
        self.train_in_cpu = train_in_cpu

    def extractRPNFeatures(self, is_train, output_dir=None, save_features=False):
        from feature_extractor_RPN import FeatureExtractorRPN
        # call class to extract rpn features:
        feature_extractor = FeatureExtractorRPN(self.cfg_path_RPN)
        features = feature_extractor(is_train, output_dir=output_dir, train_in_cpu=self.train_in_cpu, save_features=save_features)

        return features


    def extractFeatures(self, is_train, output_dir=None, save_features=False, extract_features_segmentation = False):
        from feature_extractor_detector import FeatureExtractorDetector
        # call class to extract detector features:
        feature_extractor = FeatureExtractorDetector(self.cfg_path_target_task)
        feature_extractor.falkon_rpn_models = self.falkon_rpn_models
        feature_extractor.regressors_rpn_models = self.regressors_rpn_models
        feature_extractor.stats_rpn = self.stats_rpn
        feature_extractor.falkon_detector_models = self.falkon_detector_models
        feature_extractor.regressors_detector_models = self.regressors_detector_models
        feature_extractor.stats_detector = self.stats_detector
        if self.regions_post_nms is not None:
            feature_extractor.cfg.MODEL.RPN.POST_NMS_TOP_N_TEST = self.regions_post_nms
        features = feature_extractor(is_train, output_dir=output_dir, train_in_cpu=self.train_in_cpu, save_features=save_features, extract_features_segmentation = extract_features_segmentation)

        return features

    def trainFeatureExtractor(self, output_dir=None):
        from feature_extractor_trainer import TrainerFeatureTask
        # call class to train from scratch a model on the feature task
        trainer = TrainerFeatureTask(self.cfg_path_feature_task)
        trainer(output_dir=output_dir)