import os
import sys
import torch
import math
import argparse

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'region-classifier')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'region-refiner')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'feature-extractor')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'accuracy-evaluator')))

from mrcnn_modified.data import make_data_loader
from feature_extractor import FeatureExtractor
from mrcnn_modified.config import cfg
from accuracy_evaluator import AccuracyEvaluator


from region_refiner import RegionRefiner

from py_od_utils import computeFeatStatistics_torch, normalize_COXY, falkon_models_to_cuda, load_features_classifier, load_features_regressor, load_positives_from_COXY, minibatch_positives, shuffle_negatives

import AccuracyEvaluator as ae
import copy


parser = argparse.ArgumentParser()
parser.add_argument('--icwt30', action='store_true', help='Run the iCWT experiment reported in the paper (i.e. use as TARGET-TASK the 30 objects identification task from the iCubWorld Transformations dataset). By default, run the experiment referred to as TABLE-TOP in the paper.')
parser.add_argument('--only_ood', action='store_true', help='Run only the online-object-detection experiment, i.e. without updating the RPN.')
parser.add_argument('--output_dir', action='store', type=str, help='Set experiment\'s output directory. Default directories are tabletop_experiment and icwt30_experiment, according to the dataset used.')
parser.add_argument('--save_RPN_models', action='store_true', help='Save, in the output directory, FALKON models, regressors and features statistics of the RPN.')
parser.add_argument('--save_detector_models', action='store_true', help='Save, in the output directory, FALKON models, regressors and features statistics of the detector.')
parser.add_argument('--load_RPN_models', action='store_true', help='Load, from the output directory, FALKON models, regressors and features statistics of the RPN.')
parser.add_argument('--load_detector_models', action='store_true', help='Load, from the output directory, FALKON models, regressors and features statistics of the detector.')
parser.add_argument('--load_segmentation_models', action='store_true', help='Load, from the output directory, FALKON models and features statistics of the segmentator.')
parser.add_argument('--normalize_features_regressor_detector', action='store_true', help='Normalize features for bounding box regression of the online detection.')
parser.add_argument('--CPU', action='store_true', help='Run FALKON and bbox regressors training in CPU')
parser.add_argument('--save_RPN_features', action='store_true', help='Save, in the features directory (in the output directory), RPN features.')
parser.add_argument('--save_detector_features', action='store_true', help='Save, in the features directory (in the output directory), detector\'s features.')
parser.add_argument('--load_RPN_features', action='store_true', help='Load, from the features directory (in the output directory), RPN features.')
parser.add_argument('--load_detector_features', action='store_true', help='Load, from the features directory (in the output directory), detector\'s features.')


args = parser.parse_args()

# Import the different online classifiers, depending on the device in which they must be trained
if args.CPU:
    import OnlineRegionClassifier as ocr
    import FALKONWrapper_with_centers_selection as falkon
else:
    import OnlineRegionClassifier_incore as ocr
    import FALKONWrapper_with_centers_selection_incore as falkon

# Experiment configuration

# Set chosen experiment
is_tabletop = not args.icwt30

cfg_target_task = 'configs/config_feature_extraction_segmentation_ycbv_no_ho3d_objects_with_rpn.yaml' #'configs/config_feature_extraction_segmentation_ho3d_with_rpn.yaml' #'configs/config_feature_extraction_segmentation_ycbv_with_rpn.yaml'

cfg_rpn = 'configs/config_rpn_ycbv_no_ho3d_objects.yaml' #'configs/config_rpn_ho3d.yaml' #'configs/config_rpn_ycb.yaml'
cfg_online_path = 'configs/config_online_rpn_detection_segmentation_ycbv_no_ho3d_objects.yaml' #'configs/config_online_rpn_detection_segmentation_ho3d.yaml' #'configs/config_online_rpn_detection_segmentation_ycbv.yaml'


# Set and create output directory
if args.output_dir:
    if args.output_dir.startswith('/'):
        output_dir = args.output_dir
    else:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.output_dir))
    print(args.output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


# Initialize feature extractor
feature_extractor = FeatureExtractor(cfg_target_task, cfg_rpn, train_in_cpu=args.CPU)

#feature_extractor.extractRPNFeatures(is_train=True, output_dir=output_dir, save_features=True)
#quit()

lambdas = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01]
sigmas = [1, 5, 10, 15, 20, 25, 30, 50, 100, 1000, 10000]
#lambdas = [0.001]
#sigmas = [50]
for lam in lambdas:
    for sigma in sigmas:
        print('---------------------------------------- Training with lambda %s and sigma %s ----------------------------------------' % (str(lam), str(sigma)))
        positives, negatives = load_features_classifier(features_dir = os.path.join(output_dir, 'features_RPN'))
        stats_rpn = computeFeatStatistics_torch(positives, negatives, features_dim=positives[0].size()[1], cpu_tensor=args.CPU, pos_fraction=0.8)
        
        # RPN Region Classifier initialization
        classifier = falkon.FALKONWrapper(cfg_path=cfg_online_path, is_rpn=True)
        regionClassifier = ocr.OnlineRegionClassifier(classifier, positives, negatives, stats_rpn, cfg_path=cfg_online_path, is_rpn=True)

        # Train RPN region classifier
        models_falkon_rpn = falkon_models_to_cuda(regionClassifier.trainRegionClassifier(opts={'is_rpn': True, 'lam': lam, 'sigma': sigma}, output_dir=output_dir))

        del positives, negatives, regionClassifier
        torch.cuda.empty_cache()

        # RPN Region Refiner initialization
        region_refiner = RegionRefiner(cfg_online_path, is_rpn=True)
        COXY = load_features_regressor(features_dir=os.path.join(output_dir, 'features_RPN'))

        # Train RPN region Refiner
        models_reg_rpn = region_refiner.trainRegionRefiner(normalize_COXY(COXY, stats_rpn, args.CPU), output_dir=output_dir)

        # Set trained RPN models in the pipeline
        feature_extractor.falkon_rpn_models = models_falkon_rpn
        feature_extractor.regressors_rpn_models = models_reg_rpn
        feature_extractor.stats_rpn = stats_rpn

        # Save RPN models, if requested
        if args.save_RPN_models:
            torch.save(models_falkon_rpn, os.path.join(output_dir, 'classifier_rpn'))
            torch.save(models_reg_rpn, os.path.join(output_dir, 'regressor_rpn'))
            torch.save(stats_rpn, os.path.join(output_dir, 'stats_rpn'))


        # Delete already used data
        del COXY, region_refiner
        torch.cuda.empty_cache()

        test_boxes = feature_extractor.extractFeatures(is_train=False)

