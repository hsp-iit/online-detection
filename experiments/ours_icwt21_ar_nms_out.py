import os
import sys

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'region-classifier')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'region-refiner')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'feature-extractor')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'accuracy-evaluator')))

from maskrcnn_pytorch.benchmark.data import make_data_loader
from feature_extractor import FeatureExtractor
from maskrcnn_pytorch.benchmark.config import cfg

import OnlineRegionClassifierRPNOnline as ocr
#import FALKONWrapper_with_centers_selection_logistic_loss as falkon_rpn
#import FALKONWrapper as falkon
import FALKONWrapper_with_centers_selection as falkon

from region_refiner import RegionRefiner
import torch
import math
from py_od_utils import computeFeatStatistics_torch, normalize_COXY, falkon_models_to_cuda
import AccuracyEvaluator as ae
import copy


cfg_feature_task = 'configs/config_feature_task.yaml'
is_tabletop = True
update_rpn = False
if is_tabletop:
    cfg_target_task = 'configs/config_detector_tabletop.yaml'
    cfg_rpn = 'configs/config_rpn_tabletop.yaml'
    cfg_online_path = 'configs/config_online_rpn_online_detection_tabletop.yaml'

else:
    cfg_target_task = 'configs/config_detector_icwt30.yaml'
    cfg_rpn = 'configs/config_rpn_icwt30.yaml'
    cfg_online_path = 'configs/config_online_rpn_online_detection_icwt30.yaml'



feature_extractor = FeatureExtractor(cfg_feature_task, cfg_target_task, cfg_rpn)
if update_rpn:
    # Extract RPN features for the training set
    feature_extractor.is_train = True
    negatives, positives, COXY = feature_extractor.extractRPNFeatures()
    stats_rpn = computeFeatStatistics_torch(positives, negatives,  features_dim=positives[0].size()[1])

    # ----------------------------------------------------------------------------------------
    # ------------------------------- Experiment configuration -------------------------------
    # ----------------------------------------------------------------------------------------
    # RPN Region Classifier initialization
    classifier = falkon.FALKONWrapper(cfg_path=cfg_online_path, is_rpn=True)
    regionClassifier = ocr.OnlineRegionClassifier(classifier, positives, negatives, stats_rpn, cfg_path=cfg_online_path, is_rpn=True)

    # -----------------------------------------------------------------------------------
    # --------------------------------- Training models ---------------------------------
    # -----------------------------------------------------------------------------------

    # Train RPN region classifier
    print('Region classifier test on the test set')
    models_falkon_rpn = falkon_models_to_cuda(regionClassifier.trainRegionClassifier(opts={'is_rpn': True}))
    print(models_falkon_rpn)

    # RPN Region Refiner initialization
    region_refiner = RegionRefiner(cfg_online_path, is_rpn=True)
    region_refiner.COXY = normalize_COXY(COXY, stats_rpn)

    # Train RPN region Refiner
    models_reg_rpn = region_refiner.trainRegionRefiner()


    # Setting trained RPN models in the pipeline
    feature_extractor.falkon_rpn_models = models_falkon_rpn
    feature_extractor.regressors_rpn_models = models_reg_rpn
    feature_extractor.stats_rpn = stats_rpn

# Setting trained RPN models in the pipeline
feature_extractor.falkon_rpn_models = torch.load('first_experiment/integration_tests_ep8_icwt_21_online_pipeline/classifier_rpn')
feature_extractor.regressors_rpn_models = torch.load('first_experiment/integration_tests_ep8_icwt_21_online_pipeline/regressor_rpn')
feature_extractor.stats_rpn = torch.load('first_experiment/integration_tests_ep8_icwt_21_online_pipeline/stats_rpn')

feature_extractor.is_train = False
feature_extractor.is_test = True
for i in range(10, 310, 10):
    feature_extractor.regions_post_nms = i
    feature_extractor.extractFeatures()
