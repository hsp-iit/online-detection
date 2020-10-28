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
    import OnlineRegionClassifierRPNOnline as ocr
    import FALKONWrapper_with_centers_selection as falkon
else:
    import OnlineRegionClassifierRPNOnline_incore_cpu_to_cuda as ocr
    import FALKONWrapper_with_centers_selection_incore as falkon

# Experiment configuration

# Set chosen experiment
is_tabletop = not args.icwt30
"""
# Set configuration files
if is_tabletop:
    cfg_target_task = 'configs/config_detector_tabletop.yaml'
    if not args.only_ood:
        cfg_rpn = 'configs/config_rpn_tabletop.yaml'
        cfg_online_path = 'configs/config_online_rpn_online_detection_tabletop.yaml'
    else:
        cfg_rpn = None
        cfg_online_path = 'configs/config_online_detection_tabletop.yaml'
else:
    cfg_target_task = 'configs/config_detector_icwt30.yaml'
    if not args.only_ood:
        cfg_rpn = 'configs/config_rpn_icwt30.yaml'
        cfg_online_path = 'configs/config_online_rpn_online_detection_icwt30.yaml'
    else:
        cfg_rpn = None
        cfg_online_path = 'configs/config_online_detection_icwt30.yaml'
"""

cfg_target_task = 'configs/config_segmentation_ycb.yaml'
if not args.only_ood:
    cfg_rpn = 'configs/config_rpn_ycb.yaml'
    cfg_online_path = 'configs/config_online_rpn_online_detection_tabletop.yaml'
else:
    cfg_rpn = None
    cfg_online_path = 'configs/config_online_detection_ycbv.yaml'


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
feature_extractor = FeatureExtractor(cfg_target_task, cfg_rpn, train_in_cpu=args.CPU)

# Train RPN
if not args.only_ood and not args.load_RPN_models:
    # Extract RPN features for the training set
    feature_extractor.is_train = True
    if not args.save_RPN_features and not args.load_RPN_features:
        negatives, positives, COXY = feature_extractor.extractRPNFeatures(is_train=True, output_dir=output_dir, save_features=args.save_RPN_features)
    else:
        if args.save_RPN_features:
            feature_extractor.extractRPNFeatures(is_train=True, output_dir=output_dir, save_features=args.save_RPN_features)
        positives, negatives = load_features_classifier(features_dir = os.path.join(output_dir, 'features_RPN'))
    stats_rpn = computeFeatStatistics_torch(positives, negatives, features_dim=positives[0].size()[1], cpu_tensor=args.CPU)
    
    # RPN Region Classifier initialization
    classifier = falkon.FALKONWrapper(cfg_path=cfg_online_path, is_rpn=True)
    regionClassifier = ocr.OnlineRegionClassifier(classifier, positives, negatives, stats_rpn, cfg_path=cfg_online_path, is_rpn=True)

    # Train RPN region classifier
    models_falkon_rpn = falkon_models_to_cuda(regionClassifier.trainRegionClassifier(opts={'is_rpn': True}, output_dir=output_dir))

    # RPN Region Refiner initialization
    region_refiner = RegionRefiner(cfg_online_path, is_rpn=True)
    if args.save_RPN_features or args.load_RPN_features:
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
    del negatives, positives, COXY
    torch.cuda.empty_cache()

# Load trained RPN models and set them in the pipeline, if requested
elif not args.only_ood and args.load_RPN_models:
    feature_extractor.falkon_rpn_models = torch.load(os.path.join(output_dir, 'classifier_rpn'))
    feature_extractor.regressors_rpn_models = torch.load(os.path.join(output_dir, 'regressor_rpn'))
    feature_extractor.stats_rpn = torch.load(os.path.join(output_dir, 'stats_rpn'))

elif args.only_ood and args.load_RPN_models:
    print('Unconsistency! It is not possible to run only the online object detection experiment and to load RPN models. Quitting.')
    quit()

# Load detector models if requested, else train them
if args.load_detector_models:
    model = torch.load(os.path.join(output_dir, 'classifier_detector'))
    models = torch.load(os.path.join(output_dir, 'regressor_detector'))
    stats = torch.load(os.path.join(output_dir, 'stats_detector'))
    # Detector Region Classifier initialization
    classifier = falkon.FALKONWrapper(cfg_path=cfg_online_path)
    regionClassifier = ocr.OnlineRegionClassifier(classifier, None, None, stats, cfg_path=cfg_online_path)
    region_refiner = RegionRefiner(cfg_online_path)

else:
    # Extract detector features for the train set
    if not args.save_detector_features and not args.load_detector_features:
        negatives, positives, COXY, negatives_segmentation, positives_segmentation = feature_extractor.extractFeatures(is_train=True, output_dir=output_dir, save_features=args.save_detector_features, extract_features_segmentation=False)
    """
    else:
        if args.save_detector_features:
            feature_extractor.extractFeatures(is_train=True, output_dir=output_dir, save_features=args.save_detector_features, extract_features_segmentation=True)
        positives, negatives = load_features_classifier(features_dir = os.path.join(output_dir, 'features_detector'))
    stats = computeFeatStatistics_torch(positives, negatives, features_dim=positives[0].size()[1], cpu_tensor=args.CPU)
    """
    #normalized = False
    lambdas = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01]
    sigmas = [1, 5, 10, 15, 20, 25, 30, 50, 100, 1000, 10000]
    #lambdas = [0.000001]
    #sigmas = [10]
    for lam in lambdas:
        for sigma in sigmas:
            print('---------------------------------------- Training with lambda %s and sigma %s ----------------------------------------' % (str(lam), str(sigma)))
            # Detector Region Classifier initialization
            positives, negatives = load_features_classifier(features_dir=os.path.join(output_dir, 'features_detector'))
            #positives, negatives = load_features_classifier(features_dir=os.path.join("/home/IIT.LOCAL/fceola/workspace/ws_mask/corl-code/python-online-detection/experiments/first_segmentation_ycb_pbr_coco_mask", 'features_detector'))
            #stats = computeFeatStatistics_torch(positives, negatives, features_dim=positives[0].size()[1],
            #                                    cpu_tensor=args.CPU)
            shuffle_neg = False #TODO parametrize
            if shuffle_neg:
                negatives = shuffle_negatives(negatives)
            use_only_gt_positives = False #TODO parametrize
            if not use_only_gt_positives:
                del positives
                torch.cuda.empty_cache()
                COXY = load_features_regressor(features_dir=os.path.join(output_dir, 'features_detector'))#, samples_fraction=0.2)
                positives = load_positives_from_COXY(COXY)
                del COXY
                torch.cuda.empty_cache()

            stats = computeFeatStatistics_torch(positives, negatives, features_dim=positives[0].size()[1],
                                                cpu_tensor=args.CPU, pos_fraction=0.8, neg_fraction=0.2)

            minibootstrap_positives = False
            if minibootstrap_positives: #TODO parametrize
                positives = minibatch_positives(positives, 20)

            classifier = falkon.FALKONWrapper(cfg_path=cfg_online_path)
            regionClassifier = ocr.OnlineRegionClassifier(classifier, positives, negatives, stats,
                                                          cfg_path=cfg_online_path)
            #regionClassifier.choose_positives = True
            regionClassifier.minibootstrap_positives = minibootstrap_positives
            trained=False
            try:
                #regionClassifier.normalized = normalized

                try:
                    model = falkon_models_to_cuda(regionClassifier.trainRegionClassifier(output_dir=output_dir, opts={'lam': lam, 'sigma': sigma}))
                    # Delete already used data
                    del negatives, positives, regionClassifier
                    torch.cuda.empty_cache()
                except:
                    regionClassifier.models_in_cpu = True
                    # Train detector Region Classifier
                    models_cpu = regionClassifier.trainRegionClassifier(output_dir=output_dir, opts={'lam': lam, 'sigma': sigma})
                    # Delete already used data
                    del negatives, positives, regionClassifier
                    torch.cuda.empty_cache()
                    model = falkon_models_to_cuda(models_cpu)
                trained=True

            except:
                pass

            if not trained:
                print("Training in gpu failed")
                del negatives, positives, stats, classifier, regionClassifier
                torch.cuda.empty_cache()

                # Detector Region Classifier initialization
                positives, negatives = load_features_classifier(
                    features_dir=os.path.join(output_dir, 'features_detector'))
                stats = computeFeatStatistics_torch(positives, negatives, features_dim=positives[0].size()[1],
                                                    cpu_tensor=args.CPU)

                print("Converting features to cpu tensors")
                for i in range(len(positives)):
                    positives[i] = positives[i].to('cpu')
                for i in range(len(negatives)):
                    for j in range(len(negatives[i])):
                        negatives[i][j] = negatives[i][j].to('cpu')

                classifier = falkon.FALKONWrapper(cfg_path=cfg_online_path)
                regionClassifier = ocr.OnlineRegionClassifier(classifier, positives, negatives, stats,
                                                              cfg_path=cfg_online_path)

                regionClassifier = ocr.OnlineRegionClassifier(classifier, positives, negatives, stats,
                                                              cfg_path=cfg_online_path)

                #model = falkon_models_to_cuda(
                #    regionClassifier.trainRegionClassifier(output_dir=output_dir, opts={'lam': lam, 'sigma': sigma}))
                #normalized = True
                """
                print("Training in cpu completed, converting features to cuda tensors")

                for i in range(len(positives)):
                    positives[i] = positives[i].to('cuda')
                for i in range(len(negatives)):
                    for j in range(len(negatives[i])):
                        negatives[i][j] = negatives[i][j].to('cuda')
                """
                regionClassifier.models_in_cpu = True
                models_cpu = regionClassifier.trainRegionClassifier(output_dir=output_dir, opts={'lam': lam, 'sigma': sigma})
                # Delete already used data
                del negatives, positives, regionClassifier
                torch.cuda.empty_cache()
                model = falkon_models_to_cuda(models_cpu)

            # Detector Region Refiner initialization
            region_refiner = RegionRefiner(cfg_online_path)
            if args.save_detector_features or args.load_detector_features:
                COXY = load_features_regressor(features_dir=os.path.join(output_dir, 'features_detector'))#, samples_fraction=0.2)
            if args.normalize_features_regressor_detector:
                models = region_refiner.trainRegionRefiner(normalize_COXY(COXY, stats, args.CPU), output_dir=output_dir)
            else:
                # Train Detector Region Refiner
                models = region_refiner.trainRegionRefiner(COXY, output_dir=output_dir)

            # Delete already used data
            del COXY, region_refiner
            torch.cuda.empty_cache()

            # Initialize feature extractor
            accuracy_evaluator = AccuracyEvaluator(cfg_target_task, cfg_rpn, train_in_cpu=args.CPU)
            # Set detector models in the pipeline
            accuracy_evaluator.falkon_detector_models = model
            accuracy_evaluator.regressors_detector_models = models
            accuracy_evaluator.stats_detector = stats

            #output_dir_segm = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'experiments',
            #                                          'first_segmentation_ycb_real_test'))
            #accuracy_evaluator.falkon_segmentation_models = torch.load(os.path.join(output_dir_segm, 'classifier_segmentation'))
            #accuracy_evaluator.stats_segmentation = torch.load(os.path.join(output_dir_segm, 'stats_segmentation'))

            test_boxes = accuracy_evaluator.evaluateAccuracyDetection(is_train=False, output_dir=output_dir, evaluate_segmentation=False)

    # Delete already used data
    del negatives, positives, COXY
    torch.cuda.empty_cache()

# Save detector models, if requested
if args.save_detector_models:
    torch.save(model, os.path.join(output_dir, 'classifier_detector'))
    torch.save(models, os.path.join(output_dir, 'regressor_detector'))
    torch.save(stats, os.path.join(output_dir, 'stats_detector'))

lambdas = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01]
sigmas = [1, 5, 10, 15, 20, 25, 30, 50, 100, 1000, 10000]

for lam in lambdas:
    for sigma in sigmas:
        print('---------------------------------------- Training with lambda %s and sigma %s ----------------------------------------' %(str(lam), str(sigma)))
        # Train segmentation classifiers
        positives, negatives = load_features_classifier(features_dir = os.path.join(output_dir, 'features_segmentation'), is_segm=True)
        stats_segm = computeFeatStatistics_torch(positives, negatives, features_dim=positives[0].size()[1], cpu_tensor=args.CPU)
        # Detector Region Classifier initialization
        classifier = falkon.FALKONWrapper(cfg_path=cfg_online_path)
        regionClassifier = ocr.OnlineRegionClassifier(classifier, positives, negatives, stats_segm, cfg_path=cfg_online_path)
        model_segm = falkon_models_to_cuda(regionClassifier.trainRegionClassifier(output_dir=output_dir, opts={'lam': lam, 'sigma': sigma}))

        # Initialize feature extractor
        accuracy_evaluator = AccuracyEvaluator(cfg_target_task, cfg_rpn, train_in_cpu=args.CPU)
        # Set detector models in the pipeline
        accuracy_evaluator.falkon_detector_models = model
        accuracy_evaluator.regressors_detector_models = models
        accuracy_evaluator.stats_detector = stats
    
        accuracy_evaluator.falkon_segmentation_models = model_segm
        accuracy_evaluator.stats_segmentation = stats_segm

        test_boxes = accuracy_evaluator.evaluateAccuracyDetection(is_train=False, output_dir=output_dir)
