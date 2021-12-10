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

import time

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', action='store', type=str, default='online_segmentation_experiment_ycbv', help='Set experiment\'s output directory. Default directory is segmentation_experiment_ycbv.')
parser.add_argument('--save_detector_models', action='store_true', help='Save, in the output directory, FALKON models, regressors and features statistics of the detector.')
parser.add_argument('--save_segmentation_models', action='store_true', help='Save, in the output directory, FALKON models and features statistics of the segmentator.')
parser.add_argument('--load_detector_models', action='store_true', help='Load, from the output directory, FALKON models, regressors and features statistics of the detector.')
parser.add_argument('--load_segmentation_models', action='store_true', help='Load, from the output directory, FALKON models and features statistics of the segmentator.')
parser.add_argument('--CPU', action='store_true', help='Run FALKON and bbox regressors training in CPU')
parser.add_argument('--save_detector_features', action='store_true', help='Save, in the features directory (in the output directory), detector\'s features.')
parser.add_argument('--load_detector_features', action='store_true', help='Load, from the features directory (in the output directory), detector\'s features.')
parser.add_argument('--load_segmentation_features', action='store_true', help='Load, from the features directory (in the output directory), segmentation features.')
parser.add_argument('--eval_segm_with_gt_bboxes', action='store_true', help='Evaluate segmentation accuracy, supposing that gt_bboxes are available.')
parser.add_argument('--use_only_gt_positives_detection', action='store_true', help='Consider only the ground truth bounding boxes as positive samples for the online detection.')
parser.add_argument('--sampling_ratio_segmentation', action='store', type=float, default=0.3, help='Set the fraction of positives and negatives samples to be used to train the online segmentation head, when loading features from disk, supposing that all the features were previously saved.')
parser.add_argument('--pos_fraction_feat_stats', action='store', type=float, default=0.8, help='Set the fraction of positives samples to be used to compute features statistics for data normalization')
parser.add_argument('--config_file_feature_extraction', action='store', type=str, default="config_feature_extraction_segmentation_ycbv.yaml", help='Manually set configuration file for feature extraction, by default it is config_feature_extraction_segmentation_ycbv.yaml. If the specified path is not absolute, the config file will be searched in the experiments/configs directory')
parser.add_argument('--config_file_online_detection_online_segmentation', action='store', type=str, default="config_online_detection_segmentation_ycbv.yaml", help='Manually set configuration file for online detection and segmentation, by default it is config_online_detection_segmentation_ycbv.yaml. If the specified path is not absolute, the config file will be searched in the experiments/configs directory')
parser.add_argument('--normalize_features_regressor_detector', action='store_true', help='Normalize features for bounding box regression of the online detection.')
parser.add_argument('--minibootstrap_iterations', action='store', type=int, help='Set the number of minibootstrap iterations both for rpn and detection.')

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

# Load detector models if requested, else train them
if args.load_detector_models:
    model = torch.load(os.path.join(output_dir, 'classifier_detector'))
    models = torch.load(os.path.join(output_dir, 'regressor_detector'))
    stats = torch.load(os.path.join(output_dir, 'stats_detector'))

else:
    # Extract detector features for the train set
    if not args.save_detector_features and not args.load_detector_features:
        cfg_options = {}
        if args.minibootstrap_iterations:
            cfg_options['minibootstrap_iterations'] = args.minibootstrap_iterations
        negatives, positives, COXY, negatives_segmentation, positives_segmentation = feature_extractor.extractFeatures(is_train=True, output_dir=output_dir, save_features=args.save_detector_features, extract_features_segmentation=True, use_only_gt_positives_detection=args.use_only_gt_positives_detection, cfg_options=cfg_options)
        start_of_feature_extraction_time_detection = feature_extractor.start_of_feature_extraction_time_detection
        del feature_extractor
        torch.cuda.empty_cache()

        # Detector Region Refiner initialization
        region_refiner = RegionRefiner(cfg_online_path)
        if not args.normalize_features_regressor_detector:
            models = region_refiner.trainRegionRefiner(COXY, output_dir=output_dir)

            del region_refiner
            torch.cuda.empty_cache()

        if not args.use_only_gt_positives_detection:
            positives = load_positives_from_COXY(COXY)

        # Delete already used data
        if not args.normalize_features_regressor_detector:
            del COXY
            torch.cuda.empty_cache()

        stats = computeFeatStatistics_torch(positives, negatives, features_dim=negatives[0][0].size()[1],
                                            cpu_tensor=args.CPU, pos_fraction=pos_fraction_feat_stats)

        # Detector Region Classifier initialization
        classifier = falkon.FALKONWrapper(cfg_path=cfg_online_path)
        regionClassifier = ocr.OnlineRegionClassifier(classifier, positives, negatives, stats, cfg_path=cfg_online_path)

        # Train detector Region Classifier
        model = falkon_models_to_cuda(regionClassifier.trainRegionClassifier(output_dir=output_dir))

        # Delete already used data
        del negatives, positives, regionClassifier
        torch.cuda.empty_cache()

        if args.normalize_features_regressor_detector:
            models = region_refiner.trainRegionRefiner(normalize_COXY(COXY, stats, args.CPU), output_dir=output_dir)

            del region_refiner, COXY
            torch.cuda.empty_cache()

    else:
        if args.save_detector_features:
            feature_extractor.extractFeatures(is_train=True, output_dir=output_dir, save_features=args.save_detector_features, extract_features_segmentation=True, use_only_gt_positives_detection=args.use_only_gt_positives_detection)
            del feature_extractor
            torch.cuda.empty_cache()

        if args.CPU:
            training_device = 'cpu'
        else:
            training_device = 'cuda'

        # Detector Region Refiner initialization
        region_refiner = RegionRefiner(cfg_online_path)
        # Load COXY only if regressor features do not need to be normalized or if they are required to compute positives for classification
        if not args.normalize_features_regressor_detector or not args.use_only_gt_positives_detection:
            COXY = load_features_regressor(features_dir=os.path.join(output_dir, 'features_detector'))

            # Features can be extracted in a device that does not correspond to the one used for training.
            # Convert them to the proper device.
            COXY['C'] = COXY['C'].to(training_device)
            COXY['X'] = COXY['X'].to(training_device)
            COXY['Y'] = COXY['Y'].to(training_device)

        # Train Detector Region Refiner if regressor features do not need to be normalized
        if not args.normalize_features_regressor_detector:
            models = region_refiner.trainRegionRefiner(COXY, output_dir=output_dir)

        # Delete COXY if regressors have been trained and it is not required for positives computation for classification
        if not args.normalize_features_regressor_detector and args.use_only_gt_positives_detection:
            # Delete already used data
            del region_refiner, COXY
            torch.cuda.empty_cache()

        positives, negatives = load_features_classifier(features_dir=os.path.join(output_dir, 'features_detector'), cfg_feature_extraction=cfg_target_task)

        # Load positives from COXY if required
        if not args.use_only_gt_positives_detection:
            positives = load_positives_from_COXY(COXY)
            # If regressor's features normalization is not required, delete COXY
            if not args.normalize_features_regressor_detector:
                del COXY
                torch.cuda.empty_cache()

        # Features can be extracted in a device that does not correspond to the one used for training.
        # Convert them to the proper device.
        for i in range(len(positives)):
            positives[i] = positives[i].to(training_device)
            for j in range(len(negatives[i])):
                negatives[i][j] = negatives[i][j].to(training_device)

        stats = computeFeatStatistics_torch(positives, negatives, features_dim=negatives[0][0].size()[1], cpu_tensor=args.CPU, pos_fraction=pos_fraction_feat_stats)

        # Detector Region Classifier initialization
        classifier = falkon.FALKONWrapper(cfg_path=cfg_online_path)
        regionClassifier = ocr.OnlineRegionClassifier(classifier, positives, negatives, stats, cfg_path=cfg_online_path)

        # Train detector Region Classifier
        model = falkon_models_to_cuda(regionClassifier.trainRegionClassifier(output_dir=output_dir))

        # Delete already used data
        del negatives, positives, regionClassifier
        torch.cuda.empty_cache()

        if args.normalize_features_regressor_detector and args.use_only_gt_positives_detection:
            COXY = load_features_regressor(features_dir=os.path.join(output_dir, 'features_detector'))

            # Features can be extracted in a device that does not correspond to the one used for training.
            # Convert them to the proper device.
            COXY['C'] = COXY['C'].to(training_device)
            COXY['X'] = COXY['X'].to(training_device)
            COXY['Y'] = COXY['Y'].to(training_device)

        # Train Detector Region Refiner if regressor features do not need to be normalized
        if args.normalize_features_regressor_detector:
            models = region_refiner.trainRegionRefiner(normalize_COXY(COXY, stats, args.CPU), output_dir=output_dir)
            del region_refiner, COXY
            torch.cuda.empty_cache()


# Save detector models, if requested
if args.save_detector_models:
    torch.save(model, os.path.join(output_dir, 'classifier_detector'))
    torch.save(models, os.path.join(output_dir, 'regressor_detector'))
    torch.save(stats, os.path.join(output_dir, 'stats_detector'))

if not args.load_segmentation_models:
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
    model_segm = falkon_models_to_cuda(regionClassifier.trainRegionClassifier(output_dir=output_dir))

    torch.cuda.synchronize()
    end_of_training_time = time.time()
    if not args.save_detector_features and not args.load_detector_features and not args.load_detector_models and not args.load_segmentation_features and not args.load_segmentation_models:
        total_training_time = end_of_training_time - start_of_feature_extraction_time_detection
        with open(os.path.join(output_dir, "result.txt"), "a") as fid:
            fid.write("\nTotal training time: {}min:{}s \n\n".format(int(total_training_time / 60),
                                                                     round(total_training_time % 60)))

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

test_boxes = accuracy_evaluator.evaluateAccuracyDetection(is_train=False, output_dir=output_dir, eval_segm_with_gt_bboxes=args.eval_segm_with_gt_bboxes, normalize_features_regressors=args.normalize_features_regressor_detector, evaluate_segmentation_icwt=True)
