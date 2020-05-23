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
experiment_name = 'new'
classifier = falkon.FALKONWrapper()
negative_selector = ms.MinibootstrapSelector(10, 2000, -0.9, -0.7)
regionClassifier = ocr.OnlineRegionClassifier(experiment_name, classifier, negative_selector)
accuracy_evaluator = ae.AccuracyEvaluator(score_thresh=-2, nms=0.3, detections_per_img=100, cls_agnostic_bbox_reg=True)

opts = dict()
opts['kernel_type'] = 'gauss'
opts['num_classes'] = 31
opts['output_folder'] = '/home/elisa/Repos/python-online-detection/output'

# Retrieve feature extractor (either by loading it or by training it)
print('Skip retriever feature extractor')

# Extract features for the train/val/test sets
print('Skip feature extraction')
imset_test = '/home/elisa/Data/Datasets/iCubWorld-Transformations/ImageSets/test_TASK2_30objs_manual.txt'
cfg.merge_from_file('/home/elisa/Repos/python-online-detection/experiments/Configs/first_experiment.yaml')
cfg.freeze()
dataset = make_data_loader(cfg, is_train=False, is_distributed=False, is_target_task=True)

# Train region refiner
print('Skip region refiner training')

# Start the cross validation
print('Skip cross validation')

# - Set parameters
opts['lambda'] = 0.001
opts['sigma'] = 25
opts['M'] = 1000
imset_train = '/home/elisa/Data/Datasets/iCubWorld-Transformations/ImageSets/test_TASK2_30objs_manual.txt'
# - Train region classifier
model = regionClassifier.trainRegionClassifier(imset_train, opts)
# model = torch.load('model_icub_test_TASK2_30objs_manual')
# - Test region classifier (on validation set)
print('Skip Test region classifier on validation set')

# - Test region refiner (on validation set)
print('Skip Test region refiner on validation set')

# - Save/store results
print('Skip saving model')

# Test the best model (on the test set)
scores, boxes, predictions = regionClassifier.testRegionClassifier(model, imset_test, opts)
# scores = torch.load('scores')
# boxes = torch.load('boxes')
# predictions = torch.load('predictions')

result_cls = accuracy_evaluator.evaluate(dataset.dataset, scores, boxes, predictions, opts, is_target_task=True)

# Test region refiner (on test set)
print('Skip Test region refiner on test set')

