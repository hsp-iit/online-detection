import os
import sys
import torch
import math
import argparse
import glob

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'region-classifier')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'region-refiner')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'feature-extractor')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'accuracy-evaluator')))

from feature_extractor import FeatureExtractor

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

# Set and create output directory
if args.output_dir:
    if args.output_dir.startswith('/'):
        output_dir = args.output_dir
    else:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.output_dir))
    print(args.output_dir)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

#cfg_feature_task = "configs/config_full_train_ycbv.yaml"
cfg_feature_task = "configs/config_test_feature_task_models_ycbv.yaml"



# Initialize feature extractor
feature_extractor = FeatureExtractor(cfg_path_feature_task=cfg_feature_task)

#models = glob.glob(output_dir+"/model*.pth")
models = ["/home/IIT.LOCAL/fceola/workspace/ws_mask/corl-code/python-online-detection/experiments/full_train_real_1_out_of_10_coco_pretraining/model_0022640.pth"]
for model in sorted(models):
    feature_extractor.testFeatureExtractor(output_dir=output_dir, model_to_test=model)

