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


cfg_feature_task = 'first_experiment/configs/config_feature_task_federico.yaml'
is_tabletop = True
update_rpn = False
if is_tabletop:
    cfg_target_task = 'first_experiment/configs/config_target_task_FALKON_federico_icwt_21_copy.yaml'
    cfg_rpn = 'first_experiment/configs/config_rpn_federico_icwt_21.yaml'
    cfg_online_path = 'Configs/config_federico_server_icwt_21_final.yaml'

else:
    cfg_target_task = 'first_experiment/configs/config_target_task_FALKON_federico.yaml'
    cfg_rpn = 'first_experiment/configs/config_rpn_federico.yaml'
    cfg_online_path = 'Configs/config_federico_server_icwt_30_final.yaml'



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
#feature_extractor.falkon_rpn_models = torch.load('first_experiment/integration_tests_ep8_icwt_21_online_pipeline/classifier_rpn')
#feature_extractor.regressors_rpn_models = torch.load('first_experiment/integration_tests_ep8_icwt_21_online_pipeline/regressor_rpn')
#feature_extractor.stats_rpn = torch.load('first_experiment/integration_tests_ep8_icwt_21_online_pipeline/stats_rpn')

## Extract features for the train set
feature_extractor.is_train = True
negatives, positives, COXY = feature_extractor.extractFeatures()
stats = computeFeatStatistics_torch(positives, negatives, features_dim=positives[0].size()[1])

# ----------------------------------------------------------------------------------------
# ------------------------------- Experiment configuration -------------------------------
# ----------------------------------------------------------------------------------------
# Test dataset creation
cfg.merge_from_file(cfg_target_task)
cfg.freeze()
dataset = make_data_loader(cfg, is_train=False, is_distributed=False, is_target_task=True, icwt_21_objs=is_tabletop)

# Region Classifier initialization
classifier = falkon.FALKONWrapper(cfg_path=cfg_online_path)
regionClassifier = ocr.OnlineRegionClassifier(classifier, positives, negatives, stats, cfg_path=cfg_online_path)

if not os.path.exists(regionClassifier.output_folder):
    os.mkdir(regionClassifier.output_folder)
# Accuracy evaluator initialization
accuracy_evaluator = ae.AccuracyEvaluator(cfg_online_path)
# -----------------------------------------------------------------------------------
# --------------------------------- Training models ---------------------------------
# -----------------------------------------------------------------------------------
# - Train region classifier
model = falkon_models_to_cuda(regionClassifier.trainRegionClassifier())
print(model)
# Region Refiner initialization
region_refiner = RegionRefiner(cfg_online_path)
region_refiner.COXY = normalize_COXY(COXY, stats)

# - Train region Refiner
models = region_refiner.trainRegionRefiner()

#torch.save(model, os.path.join(regionClassifier.output_folder, 'classifier_ood'))
#torch.save(models, os.path.join(regionClassifier.output_folder, 'regressor_ood'))
#torch.save(stats, os.path.join(regionClassifier.output_folder, 'stats_ood'))

# ----------------------------------------------------------------------------------
# --------------------------------- Testing models ---------------------------------
# ----------------------------------------------------------------------------------
feature_extractor.is_train = False
feature_extractor.is_test = True
test_boxes = feature_extractor.extractFeatures()
# Test the best classifier on the test set
print('Region classifier test on the test set')
predictions = regionClassifier.testRegionClassifier(model, test_boxes)

print('Region classifier predictions evaluation')
result_cls = accuracy_evaluator.evaluate(dataset.dataset, copy.deepcopy(predictions), is_target_task=True, cls_agnostic_bbox_reg=True, icwt_21_objs=is_tabletop)

# Test the best regressor on the test set
print('Region refiner test on the test set')
region_refiner.boxes = predictions
region_refiner.stats = stats
region_refiner.feat = test_boxes
refined_predictions = region_refiner.predict()

print('Region classifier predictions evaluation')
result_reg = accuracy_evaluator.evaluate(dataset.dataset, refined_predictions, is_target_task=True, cls_agnostic_bbox_reg=False, icwt_21_objs=is_tabletop)
