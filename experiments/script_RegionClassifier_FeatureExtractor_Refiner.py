import os
import sys

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'region-classifier')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'accuracy-evaluator')))
sys.path.append(os.path.abspath(os.path.join(basedir, '..', 'src', 'modules', 'feature-extractor')))
sys.path.append(os.path.abspath(os.path.join(basedir, '..', 'src', 'modules', 'region-refiner')))

import OnlineRegionClassifier as ocr
import FALKONWrapper as falkon
import MinibootstrapSelector as ms
import AccuracyEvaluator as ae
from feature_extractor import FeatureExtractor
from maskrcnn_pytorch.benchmark.data import make_data_loader
from maskrcnn_pytorch.benchmark.config import cfg
from region_refiner import RegionRefiner


# Temporary imports
import torch

# Experiment configuration
cfg.merge_from_file('Configs/first_experiment_elisa_server.yaml')
cfg.freeze()
dataset = make_data_loader(cfg, is_train=False, is_distributed=False, is_target_task=True)
cfg_path = 'Configs/config_region_classifier_elisa_server.yaml'
classifier = falkon.FALKONWrapper()
negative_selector = ms.MinibootstrapSelector(cfg_path)
regionClassifier = ocr.OnlineRegionClassifier(classifier, negative_selector, cfg_path=cfg_path)
accuracy_evaluator = ae.AccuracyEvaluator(cfg_path)
feature_extractor = FeatureExtractor('first_experiment/configs/config_feature_task_elisa_server.yaml',
                                     'first_experiment/configs/config_target_task_FALKON_elisa_server.yaml')
region_refiner = RegionRefiner('first_experiment/configs/config_region_refiner_elisa_server.yaml')

# Retrieve feature extractor (either by loading it or by training it)
print('Retrieve or train feature extractor')
try:
    feature_extractor.loadFeatureExtractor()
except OSError:
    print('Feature extractor will be trained from scratch.')
    feature_extractor.trainFeatureExtractor()

# Extract features for the train/val/test sets
print('Extract features from dataset if needed')
feature_extractor.extractFeatures()

# Train region refiner
print('')
regressors = region_refiner.trainRegionRefiner()
# torch.save(regressors, 'regressors_mask')

# Start the cross validation
print('Skip cross validation')

# - Train region classifier
model = regionClassifier.trainRegionClassifier()
# model = torch.load('model_icub_test_TASK2_30objs_manual')
# - Test region classifier (on validation set)
print('Skip Test region classifier on validation set')

# - Test region refiner (on validation set)
print('Skip Test region refiner on validation set')

# - Save/store results
print('Skip saving model')

# Test the best model (on the test set)
predictions = regionClassifier.testRegionClassifier(model)
result_cls = accuracy_evaluator.evaluate(dataset.dataset, predictions, is_target_task=True,
                                         cls_agnostic_bbox_reg=True)

# Test region refiner (on test set)
regressors.boxes = predictions
refined_predictions = region_refiner.predict()
result_reg = accuracy_evaluator.evaluate(dataset.dataset, refined_predictions, is_target_task=True,
                                         cls_agnostic_bbox_reg=False)
