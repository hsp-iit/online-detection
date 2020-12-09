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

from feature_extractor import FeatureExtractor
from accuracy_evaluator import AccuracyEvaluator
from region_refiner import RegionRefiner

from py_od_utils import computeFeatStatistics_torch, normalize_COXY, falkon_models_to_cuda, load_features_classifier, load_features_regressor, load_positives_from_COXY

parser = argparse.ArgumentParser()
parser.add_argument('--icwt30', action='store_true', help='Run the iCWT experiment reported in the paper (i.e. use as TARGET-TASK the 30 objects identification task from the iCubWorld Transformations dataset). By default, run the experiment referred to as TABLE-TOP in the paper.')
parser.add_argument('--output_dir', action='store', type=str, help='Set experiment\'s output directory. Default directories are tabletop_experiment and icwt30_experiment, according to the dataset used.')
parser.add_argument('--save_detector_models', action='store_true', help='Save, in the output directory, FALKON models, regressors and features statistics of the detector.')
parser.add_argument('--save_segmentation_models', action='store_true', help='Save, in the output directory, FALKON models and features statistics of the segmentator.')
parser.add_argument('--load_detector_models', action='store_true', help='Load, from the output directory, FALKON models, regressors and features statistics of the detector.')
parser.add_argument('--load_segmentation_models', action='store_true', help='Load, from the output directory, FALKON models and features statistics of the segmentator.')
parser.add_argument('--CPU', action='store_true', help='Run FALKON and bbox regressors training in CPU')
parser.add_argument('--save_detector_features', action='store_true', help='Save, in the features directory (in the output directory), detector\'s features.')
parser.add_argument('--load_detector_features', action='store_true', help='Load, from the features directory (in the output directory), detector\'s features.')
parser.add_argument('--load_segmentation_features', action='store_true', help='Load, from the features directory (in the output directory), detector\'s features.')
parser.add_argument('--eval_segm_with_gt_bboxes', action='store_true', help='Evaluate segmentation accuracy, supposing that gt_bboxes are available.')
parser.add_argument('--use_only_gt_positives_detection', action='store_true', help='Consider only the ground truth bounding boxes as positive samples for the online detection.')


args = parser.parse_args()

# Import the different online classifiers, depending on the device in which they must be trained
if args.CPU:
    import OnlineRegionClassifier as ocr
    import FALKONWrapper_with_centers_selection as falkon
else:
    import OnlineRegionClassifier_incore as ocr
    import FALKONWrapper_with_centers_selection_incore as falkon

# Experiment configuration
cfg_target_task = 'configs/config_segmentation_ycbv.yaml'
cfg_online_path = 'configs/config_online_detection_ycbv.yaml'

# Set and create output directory
if args.output_dir:
    if args.output_dir.startswith('/'):
        output_dir = args.output_dir
    else:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.output_dir))
    print(args.output_dir)
else:
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'segmentation_experiment'))
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


# Initialize feature extractor
feature_extractor = FeatureExtractor(cfg_target_task, train_in_cpu=args.CPU)

# Load detector models if requested, else train them
if args.load_detector_models:
    model = torch.load(os.path.join(output_dir, 'classifier_detector'))
    models = torch.load(os.path.join(output_dir, 'regressor_detector'))
    stats = torch.load(os.path.join(output_dir, 'stats_detector'))

else:
    # Extract detector features for the train set
    if not args.save_detector_features and not args.load_detector_features:
        negatives, positives, COXY, negatives_segmentation, positives_segmentation = feature_extractor.extractFeatures(is_train=True, output_dir=output_dir, save_features=args.save_detector_features, extract_features_segmentation=True, use_only_gt_positives_detection=args.use_only_gt_positives_detection)
        del feature_extractor
        torch.cuda.empty_cache()

        # Detector Region Refiner initialization
        region_refiner = RegionRefiner(cfg_online_path)
        models = region_refiner.trainRegionRefiner(COXY, output_dir=output_dir)

        del region_refiner
        torch.cuda.empty_cache()

        if not args.use_only_gt_positives_detection:
            positives = load_positives_from_COXY(COXY)

        # Delete already used data
        del COXY
        torch.cuda.empty_cache()

        stats = computeFeatStatistics_torch(positives, negatives, features_dim=negatives[0][0].size()[1],
                                            cpu_tensor=args.CPU, pos_fraction=0.8, neg_fraction=0.2)

        # Detector Region Classifier initialization
        classifier = falkon.FALKONWrapper(cfg_path=cfg_online_path)
        regionClassifier = ocr.OnlineRegionClassifier(classifier, positives, negatives, stats, cfg_path=cfg_online_path)

        # Train detector Region Classifier
        model = falkon_models_to_cuda(regionClassifier.trainRegionClassifier(output_dir=output_dir))

        # Delete already used data
        del negatives, positives, regionClassifier
        torch.cuda.empty_cache()

    else:
        if args.save_detector_features:
            feature_extractor.extractFeatures(is_train=True, output_dir=output_dir, save_features=args.save_detector_features, extract_features_segmentation=True)
            del feature_extractor
            torch.cuda.empty_cache()
        positives, negatives = load_features_classifier(features_dir = os.path.join(output_dir, 'features_detector'))

        stats = computeFeatStatistics_torch(positives, negatives, features_dim=negatives[0][0].size()[1], cpu_tensor=args.CPU, pos_fraction=0.8, neg_fraction=0.2)

        # Detector Region Classifier initialization
        classifier = falkon.FALKONWrapper(cfg_path=cfg_online_path)
        regionClassifier = ocr.OnlineRegionClassifier(classifier, positives, negatives, stats, cfg_path=cfg_online_path)

        # Train detector Region Classifier
        model = falkon_models_to_cuda(regionClassifier.trainRegionClassifier(output_dir=output_dir))

        # Delete already used data
        del negatives, positives, regionClassifier
        torch.cuda.empty_cache()

        # Detector Region Refiner initialization
        region_refiner = RegionRefiner(cfg_online_path)
        COXY = load_features_regressor(features_dir=os.path.join(output_dir, 'features_detector'))
        # Train Detector Region Refiner
        models = region_refiner.trainRegionRefiner(COXY, output_dir=output_dir)
        # Delete already used data
        del COXY
        torch.cuda.empty_cache()

# Save detector models, if requested
if args.save_detector_models:
    torch.save(model, os.path.join(output_dir, 'classifier_detector'))
    torch.save(models, os.path.join(output_dir, 'regressor_detector'))
    torch.save(stats, os.path.join(output_dir, 'stats_detector'))

if not args.load_segmentation_models:
    if args.load_segmentation_features:
        # Train segmentation classifiers
        positives_segmentation, negatives_segmentation = load_features_classifier(features_dir = os.path.join(output_dir, 'features_segmentation'), is_segm=True, sample_ratio=0.3)
    for i in range(len(positives_segmentation)):
        positives_segmentation[i] = positives_segmentation[i].to('cuda')
        negatives_segmentation[i] = [negatives_segmentation[i].to('cuda')]
    stats_segm = computeFeatStatistics_torch(positives_segmentation, negatives_segmentation, features_dim=positives_segmentation[0].size()[1], cpu_tensor=args.CPU, pos_fraction=0.8, neg_fraction=0.2)
    # Detector Region Classifier initialization
    classifier = falkon.FALKONWrapper(cfg_path=cfg_online_path, is_segmentation=True)
    regionClassifier = ocr.OnlineRegionClassifier(classifier, positives_segmentation, negatives_segmentation, stats_segm, cfg_path=cfg_online_path, is_segmentation=True)
    model_segm = falkon_models_to_cuda(regionClassifier.trainRegionClassifier(output_dir=output_dir))

    del positives_segmentation, negatives_segmentation, regionClassifier
    torch.cuda.empty_cache()
else:
    model_segm = torch.load(os.path.join(output_dir, 'classifier_segmentation'))
    stats_segm = torch.load(os.path.join(output_dir, 'stats_segmentation'))

if args.save_segmentation_models:
    torch.save(model_segm, os.path.join(output_dir, 'classifier_segmentation'))
    torch.save(stats_segm, os.path.join(output_dir, 'stats_segmentation'))

# Initialize feature extractor
accuracy_evaluator = AccuracyEvaluator(cfg_target_task, train_in_cpu=args.CPU)

# Set detector models in the pipeline
accuracy_evaluator.falkon_detector_models = model
accuracy_evaluator.regressors_detector_models = models
accuracy_evaluator.stats_detector = stats

accuracy_evaluator.falkon_segmentation_models = model_segm
accuracy_evaluator.stats_segmentation = stats_segm

test_boxes = accuracy_evaluator.evaluateAccuracyDetection(is_train=False, output_dir=output_dir, eval_segm_with_gt_bboxes=args.eval_segm_with_gt_bboxes)
