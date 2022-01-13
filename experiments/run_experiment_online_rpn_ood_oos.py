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
parser.add_argument('--output_dir', action='store', type=str, default='online_rpn_detection_segmentation_experiment_ycbv', help='Set experiment\'s output directory. Default directory is segmentation_experiment_ycbv.')
parser.add_argument('--save_RPN_detector_segmentation_models', action='store_true', help='Save, in the output directory, FALKON models, regressors and features statistics of on-line RPN, detection and segmentation.')
parser.add_argument('--load_RPN_detector_segmentation_models', action='store_true', help='Load, from the output directory, FALKON models, regressors and features statistics of on-line RPN, detection and segmentation.')
parser.add_argument('--save_RPN_detector_segmentation_features', action='store_true', help='Save, in the output directory, training features for on-line RPN, detection and segmentation.')
parser.add_argument('--load_RPN_detector_segmentation_features', action='store_true', help='Load, from the output directory, training features for on-line RPN, detection and segmentation.')
parser.add_argument('--use_only_gt_positives_detection', action='store_true', help='Consider only the ground truth bounding boxes as positive samples for the online detection.')
parser.add_argument('--sampling_ratio_segmentation', action='store', type=float, default=0.3, help='Set the fraction of positives and negatives samples to be used to train the online segmentation head, when loading features from disk, supposing that all the features were previously saved.')
parser.add_argument('--pos_fraction_feat_stats', action='store', type=float, default=0.8, help='Set the fraction of positives samples to be used to compute features statistics for data normalization')
parser.add_argument('--normalize_features_regressor_detector', action='store_true', help='Normalize features for bounding box regression of the online detection.')
parser.add_argument('--sampling_ratio_positives_detection', action='store', type=float, default=1.0, help='Set the fraction of positives samples to be used to train the online detection head, when loading the positives from COXY.')
parser.add_argument('--config_file_feature_extraction', action='store', type=str, default="config_feature_extraction_online_rpn_det_segm_ycbv.yaml", help='Manually set configuration file for feature extraction, by default it is config_feature_extraction_segmentation_ycbv.yaml. If the specified path is not absolute, the config file will be searched in the experiments/configs directory')
parser.add_argument('--config_file_online_rpn_detection_segmentation', action='store', type=str, default="config_online_rpn_detection_segmentation_ycbv.yaml", help='Manually set configuration file for online rpn, detection and segmentation, by default it is config_online_rpn_detection_segmentation_ycbv_independent_trainings.yaml. If the specified path is not absolute, the config file will be searched in the experiments/configs directory')
parser.add_argument('--minibootstrap_iterations', action='store', type=int, help='Set the number of minibootstrap iterations both for rpn and detection. Overwrites the value in the configuration file')
parser.add_argument('--CPU', action='store_true', help='Run FALKON and bbox regressors training in CPU')

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

if args.config_file_online_rpn_detection_segmentation.startswith("/"):
    cfg_online_path = args.config_file_online_rpn_detection_segmentation
else:
    cfg_online_path = os.path.abspath(os.path.join(basedir, "configs", args.config_file_online_rpn_detection_segmentation))

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

if not args.load_RPN_detector_segmentation_models:

    if not args.load_RPN_detector_segmentation_features:
        # Extract features for the training set
        cfg_options = {}
        if args.minibootstrap_iterations:
            cfg_options['minibootstrap_iterations'] = args.minibootstrap_iterations
        negatives_RPN, positives_RPN, COXY_RPN, negatives, positives, COXY, negatives_segmentation, positives_segmentation = feature_extractor.extractFeaturesRPNDetector(
            is_train=True, output_dir=output_dir, save_features=args.save_RPN_detector_segmentation_features,
            extract_features_segmentation=True,
            use_only_gt_positives_detection=args.use_only_gt_positives_detection,
            cfg_options=cfg_options
            )

        start_of_feature_extraction_time = feature_extractor.start_of_feature_extraction_time
        end_of_feature_extraction_time = feature_extractor.end_of_feature_extraction_time

        del feature_extractor
        torch.cuda.empty_cache()

    if args.load_RPN_detector_segmentation_features or args.save_RPN_detector_segmentation_features:
        positives_RPN, negatives_RPN = load_features_classifier(features_dir=os.path.join(output_dir, 'features_RPN'), cfg_feature_extraction=cfg_target_task)

    # ---------------------------------------- On-line RPN training ----------------------------------------------------
    stats_rpn = computeFeatStatistics_torch(positives_RPN, negatives_RPN, features_dim=positives_RPN[0].size()[1], cpu_tensor=args.CPU, pos_fraction=pos_fraction_feat_stats)

    # RPN Region Classifier initialization
    classifier = falkon.FALKONWrapper(cfg_path=cfg_online_path, is_rpn=True)
    regionClassifier = ocr.OnlineRegionClassifier(classifier, positives_RPN, negatives_RPN, stats_rpn, cfg_path=cfg_online_path, is_rpn=True)

    # Train RPN region classifier
    models_falkon_rpn = falkon_models_to_cuda(regionClassifier.trainRegionClassifier(opts={'is_rpn': True}, output_dir=output_dir))

    # RPN Region Refiner initialization
    region_refiner = RegionRefiner(cfg_online_path, is_rpn=True)

    if args.load_RPN_detector_segmentation_features or args.save_RPN_detector_segmentation_features:
        COXY_RPN = load_features_regressor(features_dir=os.path.join(output_dir, 'features_RPN'))

    # Train RPN region Refiner
    models_reg_rpn = region_refiner.trainRegionRefiner(normalize_COXY(COXY_RPN, stats_rpn, args.CPU), output_dir=output_dir)

    # Save RPN models, if requested
    if args.save_RPN_detector_segmentation_models:
        torch.save(models_falkon_rpn, os.path.join(output_dir, 'classifier_rpn'))
        torch.save(models_reg_rpn, os.path.join(output_dir, 'regressor_rpn'))
        torch.save(stats_rpn, os.path.join(output_dir, 'stats_rpn'))

    # Delete already used data
    del negatives_RPN, positives_RPN, COXY_RPN, regionClassifier, region_refiner, classifier
    torch.cuda.empty_cache()

    # ---------------------------------------- On-line detection training ----------------------------------------------
    if not args.load_RPN_detector_segmentation_features and not args.save_RPN_detector_segmentation_features:
        # Detector Region Refiner initialization
        region_refiner = RegionRefiner(cfg_online_path)
        if not args.normalize_features_regressor_detector:
            models = region_refiner.trainRegionRefiner(COXY, output_dir=output_dir)

            del region_refiner
            torch.cuda.empty_cache()

        if not args.use_only_gt_positives_detection:
            positives = load_positives_from_COXY(COXY, samples_fraction=args.sampling_ratio_positives_detection)

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
        del negatives, positives, regionClassifier, classifier
        torch.cuda.empty_cache()

        if args.normalize_features_regressor_detector:
            models = region_refiner.trainRegionRefiner(normalize_COXY(COXY, stats, args.CPU), output_dir=output_dir)

            del region_refiner, COXY
            torch.cuda.empty_cache()
    else:
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
            positives = load_positives_from_COXY(COXY, samples_fraction=args.sampling_ratio_positives_detection)
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
    if args.save_RPN_detector_segmentation_models:
        torch.save(model, os.path.join(output_dir, 'classifier_detector'))
        torch.save(models, os.path.join(output_dir, 'regressor_detector'))
        torch.save(stats, os.path.join(output_dir, 'stats_detector'))

    # ---------------------------------------- On-line segmentation training -------------------------------------------

    if args.load_RPN_detector_segmentation_features or args.save_RPN_detector_segmentation_features:
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

    del positives_segmentation, negatives_segmentation, regionClassifier
    torch.cuda.empty_cache()

    # Save segmentation models, if requested
    if args.save_RPN_detector_segmentation_models:
        torch.save(model_segm, os.path.join(output_dir, 'classifier_segmentation'))
        torch.save(stats_segm, os.path.join(output_dir, 'stats_segmentation'))

    if not args.load_RPN_detector_segmentation_models and not args.load_RPN_detector_segmentation_features:
        torch.cuda.synchronize()
        current_time = time.time()
        total_time = current_time - start_of_feature_extraction_time
        with open(os.path.join(output_dir, "result.txt"), "a") as fid:
            fid.write("\nTotal training time: {}min:{}s \n".format(int(total_time/60), round(total_time%60)))
        tr_time = current_time - end_of_feature_extraction_time
        with open(os.path.join(output_dir, "result.txt"), "a") as fid:
            fid.write("Training time for the online modules: {}min:{}s \n\n".format(int(tr_time/60), round(tr_time%60)))

# Load trained models and set them in the pipeline, if requested
else:
    models_falkon_rpn = torch.load(os.path.join(output_dir, 'classifier_rpn'))
    models_reg_rpn = torch.load(os.path.join(output_dir, 'regressor_rpn'))
    stats_rpn = torch.load(os.path.join(output_dir, 'stats_rpn'))
    model = torch.load(os.path.join(output_dir, 'classifier_detector'))
    models = torch.load(os.path.join(output_dir, 'regressor_detector'))
    stats = torch.load(os.path.join(output_dir, 'stats_detector'))
    model_segm = torch.load(os.path.join(output_dir, 'classifier_segmentation'))
    stats_segm = torch.load(os.path.join(output_dir, 'stats_segmentation'))

# Initialize feature extractor
accuracy_evaluator = AccuracyEvaluator(cfg_target_task, train_in_cpu=args.CPU)

accuracy_evaluator.falkon_rpn_models = models_falkon_rpn
accuracy_evaluator.regressors_rpn_models = models_reg_rpn
accuracy_evaluator.stats_rpn = stats_rpn

# Set detector models in the pipeline
accuracy_evaluator.falkon_detector_models = model
accuracy_evaluator.regressors_detector_models = models
accuracy_evaluator.stats_detector = stats

accuracy_evaluator.falkon_segmentation_models = model_segm
accuracy_evaluator.stats_segmentation = stats_segm

test_boxes = accuracy_evaluator.evaluateAccuracyDetection(is_train=False, output_dir=output_dir, eval_segm_with_gt_bboxes=False, normalize_features_regressors=args.normalize_features_regressor_detector, evaluate_segmentation_icwt=True)
