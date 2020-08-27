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
import FALKONWrapper_with_centers_selection as falkon

from region_refiner import RegionRefiner
import torch
import math
from py_od_utils import computeFeatStatistics_torch, normalize_COXY, falkon_models_to_cuda
import AccuracyEvaluator as ae
import copy

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--icwt30', action='store_true', help='Run the experiment on the icwt30 dataset.')
parser.add_argument('--only_ood', action='store_true', help='Run only the online-object-detection experiment, i.e. without updating the RPN.')
parser.add_argument('--output_dir', action='store', type=str, help='Set experiment\'s output directory. Default directories are tabletop_experiment and icwt30_experiment, according to the dataset used.')
parser.add_argument('--save_RPN_models', action='store_true', help='Save, in the output directory, FALKON models, regressors and features statistics of the RPN.')
parser.add_argument('--save_detector_models', action='store_true', help='Save, in the output directory, FALKON models, regressors and features statistics of the detector.')
parser.add_argument('--load_RPN_models', action='store_true', help='Load, from the output directory, FALKON models, regressors and features statistics of the RPN.')
parser.add_argument('--load_detector_models', action='store_true', help='Save, from the output directory, FALKON models, regressors and features statistics of the detector.')
parser.add_argument('--normalize_features_regressor_detector', action='store_true', help='Normalize features for bounding box regression of the online detection.')

args = parser.parse_args()

# Experiment configuration

# Set configuration files
cfg_feature_task = 'configs/config_feature_task.yaml'
is_tabletop = not args.icwt30
if is_tabletop:
    cfg_target_task = 'configs/config_detector_tabletop.yaml'
    cfg_rpn = 'configs/config_rpn_tabletop.yaml'
    cfg_online_path = 'configs/config_online_rpn_online_detection_tabletop.yaml'
else:
    cfg_target_task = 'configs/config_detector_icwt30.yaml'
    cfg_rpn = 'configs/config_rpn_icwt30.yaml'
    cfg_online_path = 'configs/config_online_rpn_online_detection_icwt30.yaml'

# Set if RPN must be updates
update_rpn = not args.only_ood

# Set and create output directory
if args.output_dir:
    if args.output_dir.startswith('/'):
        output_dir = args.output_dir
    else:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.output_dir))
    print(args.output_dir)
else:
    if is_tabletop:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tabletop_experiment'))
    else:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'icwt30_experiment'))
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


# Initialize feature extractor
feature_extractor = FeatureExtractor(cfg_feature_task, cfg_target_task, cfg_rpn)

if update_rpn:
    # Extract RPN features for the training set
    feature_extractor.is_train = True
    negatives, positives, COXY = feature_extractor.extractRPNFeatures()
    stats_rpn = computeFeatStatistics_torch(positives, negatives,  features_dim=positives[0].size()[1])

    # RPN Region Classifier initialization
    classifier = falkon.FALKONWrapper(cfg_path=cfg_online_path, is_rpn=True)
    regionClassifier = ocr.OnlineRegionClassifier(classifier, positives, negatives, stats_rpn, cfg_path=cfg_online_path, is_rpn=True)

    # Train RPN region classifier
    models_falkon_rpn = falkon_models_to_cuda(regionClassifier.trainRegionClassifier(opts={'is_rpn': True}))

    # RPN Region Refiner initialization
    region_refiner = RegionRefiner(cfg_online_path, is_rpn=True)
    region_refiner.COXY = normalize_COXY(COXY, stats_rpn)

    # Train RPN region Refiner
    models_reg_rpn = region_refiner.trainRegionRefiner()

    # Set trained RPN models in the pipeline
    feature_extractor.falkon_rpn_models = models_falkon_rpn
    feature_extractor.regressors_rpn_models = models_reg_rpn
    feature_extractor.stats_rpn = stats_rpn

    # Save RPN models, if requested
    if args.save_RPN_models:
        torch.save(models_falkon_rpn, os.path.join(output_dir, 'classifier_rpn'))
        torch.save(models_reg_rpn, os.path.join(output_dir, 'regressor_rpn'))
        torch.save(stats_rpn, os.path.join(output_dir, 'stats_rpn'))

# Load trained RPN models and set them in the pipeline, if requested
if args.load_RPN_models:
    feature_extractor.falkon_rpn_models = torch.load(os.path.join(output_dir, 'classifier_rpn'))
    feature_extractor.regressors_rpn_models = torch.load(os.path.join(output_dir, 'regressor_rpn'))
    feature_extractor.stats_rpn = torch.load(os.path.join(output_dir, 'stats_rpn'))

# Load detector models if requested, else train them
if args.load_detector_models:
    model = torch.load(os.path.join(output_dir, 'classifier_detector'))
    models = torch.load(os.path.join(output_dir, 'regressor_detector'))
    stats = torch.load(os.path.join(output_dir, 'stats_detector'))
else:
    # Extract detector features for the train set
    feature_extractor.is_train = True
    negatives, positives, COXY = feature_extractor.extractFeatures()
    stats = computeFeatStatistics_torch(positives, negatives, features_dim=positives[0].size()[1])

    # Detector Region Classifier initialization
    classifier = falkon.FALKONWrapper(cfg_path=cfg_online_path)
    regionClassifier = ocr.OnlineRegionClassifier(classifier, positives, negatives, stats, cfg_path=cfg_online_path)

    # Train detector Region Classifier
    model = falkon_models_to_cuda(regionClassifier.trainRegionClassifier())

    # Detector Region Refiner initialization
    region_refiner = RegionRefiner(cfg_online_path)
    if args.normalize_features_regressor_detector:
        region_refiner.COXY = normalize_COXY(COXY, stats)
    else:
        region_refiner.COXY = COXY

    # Train Detector Region Refiner
    models = region_refiner.trainRegionRefiner()

# Save detector models, if requested
if args.save_detector_models:
    torch.save(model, os.path.join(output_dir, 'classifier_detector'))
    torch.save(models, os.path.join(output_dir, 'regressor_detector'))
    torch.save(stats, os.path.join(output_dir, 'stats_detector'))

# Test models
feature_extractor.is_train = False
feature_extractor.is_test = True
print('Extracting features for the test set')
test_boxes = feature_extractor.extractFeatures()

# Compute classification predictions
print('Computing classification predictions')
predictions = regionClassifier.testRegionClassifier(model, test_boxes)

# Refine predictions with the region refiners
print('Refining predictions with bounding box regressors')
region_refiner.boxes = predictions
region_refiner.stats = stats
region_refiner.feat = test_boxes
region_refiner.normalize_features = args.normalize_features_regressor_detector
refined_predictions = region_refiner.predict()

# Test dataset creation for accuracy evaluation
print('Computing test dataset for accuracy evaluation')
cfg.merge_from_file(cfg_target_task)
cfg.freeze()
dataset = make_data_loader(cfg, is_train=False, is_distributed=False, is_target_task=True, icwt_21_objs=is_tabletop)

# Detector Accuracy evaluator initialization
print('Accuracy evaluator initialization')
accuracy_evaluator = ae.AccuracyEvaluator(cfg_online_path, output_dir)

# Compute accuracy
print('Computing accuracy')
result_reg = accuracy_evaluator.evaluate(dataset.dataset, refined_predictions, is_target_task=True, cls_agnostic_bbox_reg=False, icwt_21_objs=is_tabletop)
