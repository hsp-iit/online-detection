import os
import sys
import torch
import argparse

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'region-classifier')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'region-refiner')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'feature-extractor')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'accuracy-evaluator')))

from feature_extractor import FeatureExtractor
from accuracy_evaluator import AccuracyEvaluator
from region_refiner import RegionRefiner

from py_od_utils import computeFeatStatistics_torch, normalize_COXY, falkon_models_to_cuda, load_features_classifier, load_features_regressor, load_positives_from_COXY

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', action='store', type=str, default='online_segmentation_experiment_ycbv', help='Set experiment\'s output directory. Default directory is segmentation_experiment_ycbv.')
parser.add_argument('--save_detector_models', action='store_true', help='Save, in the output directory, FALKON models, regressors and features statistics of the detector.')
parser.add_argument('--save_segmentation_models', action='store_true', help='Save, in the output directory, FALKON models and features statistics of the segmentator.')
parser.add_argument('--load_RPN_models', action='store_true', help='Load, from the output directory, FALKON models, regressors and features statistics of the RPN.')
parser.add_argument('--load_detector_models', action='store_true', help='Load, from the output directory, FALKON models, regressors and features statistics of the detector.')
parser.add_argument('--load_segmentation_models', action='store_true', help='Load, from the output directory, FALKON models and features statistics of the segmentator.')
parser.add_argument('--CPU', action='store_true', help='Run FALKON and bbox regressors training in CPU')
parser.add_argument('--save_detector_features', action='store_true', help='Save, in the features directory (in the output directory), detector\'s features.')
parser.add_argument('--load_detector_features', action='store_true', help='Load, from the features directory (in the output directory), detector\'s features.')
parser.add_argument('--load_segmentation_features', action='store_true', help='Load, from the features directory (in the output directory), detector\'s features.')
parser.add_argument('--eval_segm_with_gt_bboxes', action='store_true', help='Evaluate segmentation accuracy, supposing that gt_bboxes are available.')
parser.add_argument('--use_only_gt_positives_detection', action='store_true', help='Consider only the ground truth bounding boxes as positive samples for the online detection.')
parser.add_argument('--sampling_ratio_segmentation', action='store', type=float, default=0.3, help='Set the fraction of positives and negatives samples to be used to train the online segmentation head, when loading features from disk, supposing that all the features were previously saved.')
parser.add_argument('--pos_fraction_feat_stats', action='store', type=float, default=0.8, help='Set the fraction of positives samples to be used to compute features statistics for data normalization')
parser.add_argument('--config_file_feature_extraction', action='store', type=str, default="config_feature_extraction_segmentation_ycbv.yaml", help='Manually set configuration file for feature extraction, by default it is config_feature_extraction_segmentation_ycbv.yaml. If the specified path is not absolute, the config file will be searched in the experiments/configs directory')
parser.add_argument('--config_file_online_detection_online_segmentation', action='store', type=str, default="config_online_detection_segmentation_ycbv.yaml", help='Manually set configuration file for online detection and segmentation, by default it is config_online_detection_segmentation_ycbv.yaml. If the specified path is not absolute, the config file will be searched in the experiments/configs directory')
parser.add_argument('--normalize_features_regressor_detector', action='store_true', help='Normalize features for bounding box regression of the online detection.')


args = parser.parse_args()

# Import the different online classifiers, depending on the device in which they must be trained
if args.CPU:
    import OnlineRegionClassifier as ocr
    import FALKONWrapper_with_centers_selection as falkon
else:
    import OnlineRegionClassifier_incore as ocr
    import FALKONWrapper_with_centers_selection_incore as falkon

# Experiment configuration
if args.config_file_feature_extraction.startswith("/"):
    cfg_target_task = args.config_file_feature_extraction
else:
    cfg_target_task = os.path.abspath(os.path.join(basedir, "configs", args.config_file_feature_extraction))

if args.config_file_online_detection_online_segmentation.startswith("/"):
    cfg_online_path = args.config_file_online_detection_online_segmentation
else:
    cfg_online_path = os.path.abspath(os.path.join(basedir, "configs", args.config_file_online_detection_online_segmentation))

if args.pos_fraction_feat_stats <= 1 and args.pos_fraction_feat_stats >= 0:
    pos_fraction_feat_stats = args.pos_fraction_feat_stats
else:
    pos_fraction_feat_stats = None

# Set and create output directory
if args.output_dir.startswith('/'):
    output_dir = args.output_dir
else:
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.output_dir))
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Initialize feature extractor
feature_extractor = FeatureExtractor(cfg_target_task, train_in_cpu=args.CPU)

if args.load_RPN_models:
    feature_extractor.falkon_rpn_models = torch.load(os.path.join(output_dir, 'classifier_rpn'))
    feature_extractor.regressors_rpn_models = torch.load(os.path.join(output_dir, 'regressor_rpn'))
    feature_extractor.stats_rpn = torch.load(os.path.join(output_dir, 'stats_rpn'))

model = torch.load(os.path.join(output_dir, 'classifier_detector'))
models = torch.load(os.path.join(output_dir, 'regressor_detector'))
stats = torch.load(os.path.join(output_dir, 'stats_detector'))

#lambdas = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01]
lambdas = [0.000001, 0.00001, 0.0001, 0.001, 0.01]
sigmas = [1, 5, 10, 15, 20, 25, 30, 50, 100, 1000, 10000]
#lambdas = [0.001]
#sigmas = [50]
for lam in lambdas:
    for sigma in sigmas:
        print('---------------------------------------- Training with lambda %s and sigma %s ----------------------------------------' % (str(lam), str(sigma)))

        try:
            # Manage all the cases in which features must be loaded
            if args.load_segmentation_features or args.save_detector_features or args.load_detector_features or args.load_detector_models:
                # Train segmentation classifiers
                positives_segmentation, negatives_segmentation = load_features_classifier(features_dir=os.path.join(output_dir, 'features_segmentation'), is_segm=True, sample_ratio=args.sampling_ratio_segmentation)
            if args.CPU:
                training_device = 'cpu'
            else:
                training_device = 'cuda'
            # Features can be extracted in a device that does not correspond to the one used for training.
            # Convert them to the proper device.
            for i in range(len(positives_segmentation)):
                positives_segmentation[i] = positives_segmentation[i].to(training_device)
                negatives_segmentation[i] = [negatives_segmentation[i].to(training_device)]
            stats_segm = computeFeatStatistics_torch(positives_segmentation, negatives_segmentation, features_dim=positives_segmentation[0].size()[1], cpu_tensor=args.CPU, pos_fraction=pos_fraction_feat_stats)
            # Per-pixel classifier initialization
            classifier = falkon.FALKONWrapper(cfg_path=cfg_online_path, is_segmentation=True)
            regionClassifier = ocr.OnlineRegionClassifier(classifier, positives_segmentation, negatives_segmentation, stats_segm, cfg_path=cfg_online_path, is_segmentation=True)
            model_segm = falkon_models_to_cuda(regionClassifier.trainRegionClassifier(opts={'lam': lam, 'sigma': sigma},output_dir=output_dir))

            del positives_segmentation, negatives_segmentation, regionClassifier
            torch.cuda.empty_cache()

            # Initialize feature extractor
            accuracy_evaluator = AccuracyEvaluator(cfg_target_task, train_in_cpu=args.CPU)

            if args.load_RPN_models:
                accuracy_evaluator.falkon_rpn_models = torch.load(os.path.join(output_dir, 'classifier_rpn'))
                accuracy_evaluator.regressors_rpn_models = torch.load(os.path.join(output_dir, 'regressor_rpn'))
                accuracy_evaluator.stats_rpn = torch.load(os.path.join(output_dir, 'stats_rpn'))

            # Set detector models in the pipeline
            accuracy_evaluator.falkon_detector_models = model
            accuracy_evaluator.regressors_detector_models = models
            accuracy_evaluator.stats_detector = stats

            accuracy_evaluator.falkon_segmentation_models = model_segm
            accuracy_evaluator.stats_segmentation = stats_segm

            test_boxes = accuracy_evaluator.evaluateAccuracyDetection(is_train=False, output_dir=output_dir, eval_segm_with_gt_bboxes=args.eval_segm_with_gt_bboxes, normalize_features_regressors=args.normalize_features_regressor_detector, evaluate_segmentation_icwt=True)
        except:
            continue
