import os
import sys

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'region-classifier')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'accuracy-evaluator')))

import OnlineRegionClassifier as ocr
import FALKONWrapper as falkon
import MinibootstrapSelector as ms
import AccuracyEvaluator as ae
from maskrcnn_pytorch.benchmark.data import make_data_loader
from maskrcnn_pytorch.benchmark.config import cfg


# Temporary imports
import torch
# Experiment configuration
cfg_path = '/home/elisa/Repos/python-online-detection/experiments/Configs/config_region_classifier_elisa.yaml'
classifier = falkon.FALKONWrapper()
negative_selector = ms.MinibootstrapSelector(cfg_path)
regionClassifier = ocr.OnlineRegionClassifier(classifier, negative_selector, cfg_path=cfg_path)
accuracy_evaluator = ae.AccuracyEvaluator(cfg_path)

# Retrieve feature extractor (either by loading it or by training it)
print('Skip retriever feature extractor')

# Extract features for the train/val/test sets
print('Skip feature extraction')
cfg.merge_from_file('/home/elisa/Repos/python-online-detection/experiments/Configs/first_experiment_elisa.yaml')
cfg.freeze()
dataset = make_data_loader(cfg, is_train=False, is_distributed=False, is_target_task=True)

# Train region refiner
print('Skip region refiner training')

# Start the cross validation
print('Skip cross validation')

# - Train region classifier
model = regionClassifier.trainRegionClassifier()
#model = torch.load('model_icub_test_TASK2_30objs_manual')
# - Test region classifier (on validation set)
print('Skip Test region classifier on validation set')

# - Test region refiner (on validation set)
print('Skip Test region refiner on validation set')

# - Save/store results
print('Skip saving model')

# Test the best model (on the test set)
predictions = regionClassifier.testRegionClassifier(model)

result_cls = accuracy_evaluator.evaluate(dataset.dataset, predictions, is_target_task=True, cls_agnostic_bbox_reg=True)

# Test region refiner (on test set)
print('Skip Test region refiner on test set')
