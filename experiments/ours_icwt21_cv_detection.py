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
#import FALKONWrapper_with_centers_selection_logistic_loss as falkon_rpn
#import FALKONWrapper as falkon
import FALKONWrapper_with_centers_selection as falkon

from region_refiner import RegionRefiner
import torch
import math
from py_od_utils import computeFeatStatistics_torch, normalize_COXY, falkon_models_to_cuda
import AccuracyEvaluator as ae
import copy


cfg_feature_task = 'first_experiment/configs/config_feature_task_federico.yaml'
is_tabletop = False
update_rpn = False
if is_tabletop:
    cfg_target_task = 'first_experiment/configs/config_target_task_FALKON_federico_icwt_21_copy.yaml'
    cfg_rpn = 'first_experiment/configs/config_rpn_federico_icwt_21.yaml'
    cfg_online_path = 'Configs/config_federico_server_icwt_21_final.yaml'

else:
    cfg_target_task = 'first_experiment/configs/config_target_task_FALKON_federico.yaml'
    cfg_rpn = 'first_experiment/configs/config_rpn_federico.yaml'
    cfg_online_path = 'Configs/config_federico_server_icwt_30_final.yaml'



feature_extractor = FeatureExtractor(cfg_feature_task, cfg_target_task, cfg_rpn)
if update_rpn:
    # Extract RPN features for the training set
    feature_extractor.is_train = True
    negatives, positives, COXY = feature_extractor.extractRPNFeatures()
    stats_rpn = computeFeatStatistics_torch(positives, negatives,  features_dim=positives[0].size()[1])

    # ----------------------------------------------------------------------------------------
    # ------------------------------- Experiment configuration -------------------------------
    # ----------------------------------------------------------------------------------------
    # RPN Region Classifier initialization
    classifier = falkon.FALKONWrapper(cfg_path=cfg_online_path, is_rpn=True)
    regionClassifier = ocr.OnlineRegionClassifier(classifier, positives, negatives, stats_rpn, cfg_path=cfg_online_path, is_rpn=True)

    # -----------------------------------------------------------------------------------
    # --------------------------------- Training models ---------------------------------
    # -----------------------------------------------------------------------------------

    # Train RPN region classifier
    print('Region classifier test on the test set')
    models_falkon_rpn = falkon_models_to_cuda(regionClassifier.trainRegionClassifier(opts={'is_rpn': True}))
    print(models_falkon_rpn)

    # RPN Region Refiner initialization
    region_refiner = RegionRefiner(cfg_online_path, is_rpn=True)
    region_refiner.COXY = normalize_COXY(COXY, stats_rpn)

    # Train RPN region Refiner
    models_reg_rpn = region_refiner.trainRegionRefiner()


    # Setting trained RPN models in the pipeline
    feature_extractor.falkon_rpn_models = models_falkon_rpn
    feature_extractor.regressors_rpn_models = models_reg_rpn
    feature_extractor.stats_rpn = stats_rpn

    if not os.path.exists(regionClassifier.output_folder):
        os.mkdir(regionClassifier.output_folder)
    torch.save(models_falkon_rpn, os.path.join(regionClassifier.output_folder, 'classifier_rpn'))
    torch.save(models_reg_rpn, os.path.join(regionClassifier.output_folder, 'regressor_rpn'))
    torch.save(stats_rpn, os.path.join(regionClassifier.output_folder, 'stats_rpn'))

## Extract features for the train set
feature_extractor.is_train = True
feature_extractor.is_test = False
negatives, positives, COXY = feature_extractor.extractFeatures()
stats = computeFeatStatistics_torch(positives, negatives, features_dim=positives[0].size()[1])
models = None
# ----------------------------------------------------------------------------------------
# ------------------------------- Experiment configuration -------------------------------
# ----------------------------------------------------------------------------------------
# Test dataset creation
cfg.merge_from_file(cfg_target_task)
cfg.freeze()
dataset = make_data_loader(cfg, is_train=False, is_distributed=False, is_target_task=True, icwt_21_objs=is_tabletop)

# Region Classifier initialization
classifier = falkon.FALKONWrapper(cfg_path=cfg_online_path)
regionClassifier = ocr.OnlineRegionClassifier(classifier, positives, negatives, stats, cfg_path=cfg_online_path)

if not os.path.exists(regionClassifier.output_folder):
    os.mkdir(regionClassifier.output_folder)
# Accuracy evaluator initialization
accuracy_evaluator = ae.AccuracyEvaluator(cfg_online_path)


feature_extractor.is_train = False
feature_extractor.is_test = True
test_boxes = feature_extractor.extractFeatures()

print('Start cross validation')

lambdas = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01]
sigmas = [1, 5, 10, 15, 20, 25, 30, 50, 100, 1000, 10000]
for lam in lambdas:
    for sigma in sigmas:
        print('---------------------------------------- Training with lambda %s and sigma %s ----------------------------------------' %(str(lam), str(sigma)))
        with open(os.path.join(regionClassifier.output_folder, "result.txt"), "a") as fid:
            fid.write('---------------------------------------- Training with lambda %s and sigma %s ----------------------------------------\n' %(str(lam), str(sigma)))
        # -----------------------------------------------------------------------------------
        # --------------------------------- Training models ---------------------------------
        # -----------------------------------------------------------------------------------
        # - Train region classifier
        model = falkon_models_to_cuda(regionClassifier.trainRegionClassifier(opts={'lam': lam, 'sigma': sigma}))
        print(model)

        # - Train region Refiner
        if models is None:
            # Region Refiner initialization
            region_refiner = RegionRefiner(cfg_online_path)
            #region_refiner.COXY = normalize_COXY(COXY, stats)
            # TODO decide if using normalization for regressors
            region_refiner.COXY = COXY
            models = region_refiner.trainRegionRefiner()

        # ----------------------------------------------------------------------------------
        # --------------------------------- Testing models ---------------------------------
        # ----------------------------------------------------------------------------------
        # Test the best classifier on the test set
        print('Region classifier test on the test set')
        predictions = regionClassifier.testRegionClassifier(model, copy.deepcopy(test_boxes))

        #print('Region classifier predictions evaluation')
        #result_cls = accuracy_evaluator.evaluate(dataset.dataset, copy.deepcopy(predictions), is_target_task=True, cls_agnostic_bbox_reg=True, icwt_21_objs=is_tabletop)

        # Test the best regressor on the test set
        print('Region refiner test on the test set')
        region_refiner.boxes = predictions
        region_refiner.stats = stats
        region_refiner.feat = test_boxes
        refined_predictions = region_refiner.predict()

        print('Region classifier predictions evaluation')
        result_reg = accuracy_evaluator.evaluate(dataset.dataset, refined_predictions, is_target_task=True, cls_agnostic_bbox_reg=False, icwt_21_objs=is_tabletop)
