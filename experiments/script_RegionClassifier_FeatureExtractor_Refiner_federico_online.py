import os
import sys

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'region-classifier')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'accuracy-evaluator')))
sys.path.append(os.path.abspath(os.path.join(basedir, '..', 'src', 'modules', 'feature-extractor')))
sys.path.append(os.path.abspath(os.path.join(basedir, '..', 'src', 'modules', 'region-refiner')))
sys.path.append(os.path.abspath(os.path.join(basedir, '..', 'src')))

from maskrcnn_pytorch.benchmark.data import make_data_loader
from feature_extractor import FeatureExtractor
from maskrcnn_pytorch.benchmark.config import cfg

import OnlineRegionClassifierRPNOnline as ocr
import AccuracyEvaluator as ae
import FALKONWrapper as falkon
#import FALKONWrapper_with_centers_selection_logistic_loss as falkon
from region_refiner import RegionRefiner
import torch
import math
import copy
from py_od_utils import computeFeatStatistics_torch


feature_extractor = FeatureExtractor('first_experiment/configs/config_feature_task_federico.yaml', 'first_experiment/configs/config_target_task_FALKON_federico_icwt_21_copy.yaml')#'first_experiment/configs/config_target_task_FALKON_federico.yaml') ## # #'configs/config_target_task_FALKON_COCO.yaml')# # # # # # # #'configs/config_target_task_FALKON_COCO.yaml')#  #'configs/config_target_task_FALKON_federico.yaml') #
"""
## Retrieve feature extractor (either by loading it or by training it)
try:
    feature_extractor.loadFeatureExtractor()
except OSError:
    print('Feature extractor will be trained from scratch.')
    feature_extractor.trainFeatureExtractor()
"""
## Extract features for the train/val/test sets
feature_extractor.is_train = True
negatives, positives, COXY = feature_extractor.extractFeatures()
stats = computeFeatStatistics_torch(positives, negatives,  features_dim=2048)

# ----------------------------------------------------------------------------------------
# ------------------------------- Experiment configuration -------------------------------
# ----------------------------------------------------------------------------------------
cfg_online_path = 'Configs/config_federico_server_icwt_21.yaml' #'Configs/config_federico_server_copy.yaml' # # #'Configs/config_federico_server_icwt_21.yaml' # # # # #'Configs/config_federico_server_copy_rpn.yaml' 'Configs/config_federico_server_copy_coco.yaml' #
cfg_target_path = 'Configs/config_target_task_federico_server_icwt21.yaml' #'Configs/config_target_task_federico_server.yaml' # #'Configs/config_target_task_federico_server_icwt21.yaml' 'Configs/config_target_task_federico_server_coco.yaml' #
cfg_feature_path = 'Configs/config_feature_task_federico_server.yaml'


# TODO do this more clever
if '21' in cfg_online_path:
    icwt_21_objs = True
else:
    icwt_21_objs = False

# Test dataset creation
cfg.merge_from_file(cfg_target_path)
cfg.freeze()
dataset = make_data_loader(cfg, is_train=False, is_distributed=False, is_target_task=True, icwt_21_objs=icwt_21_objs)

# Region Classifier initialization
classifier = falkon.FALKONWrapper(cfg_path=cfg_online_path)
regionClassifier = ocr.OnlineRegionClassifier(classifier, positives, negatives, stats, cfg_path=cfg_online_path)
# Feature extraction initialization
# feature_extractor = FeatureExtractor(cfg_feature_path, cfg_target_path)

if not os.path.exists(regionClassifier.output_folder):
    os.mkdir(regionClassifier.output_folder)
# Accuracy evaluator initialization
accuracy_evaluator = ae.AccuracyEvaluator(cfg_online_path)
# -----------------------------------------------------------------------------------
# --------------------------------- Training models ---------------------------------
# -----------------------------------------------------------------------------------
# Train region refiner
#print('Train region refiner')
#regressors = region_refiner.trainRegionRefiner()

# Start the cross validation
print('Start cross validation')

#lambdas = [0.0001, 0.00001, 0.000001, 0.0000001, 0.001]
#sigmas = [5, 10, 15, 20, 25, 30, 50, 100, 1, 1000, 10000]
lambdas =  [0.001] #[0.001]#[0.0001]#[0.0001]#[0.001]#, 0.00001, 0.000001, 0.0000001, 0.001]
sigmas = [15] #[20]#[15] #[25]#[5]#[20]#, 10, 15, 20, 25, 30, 50, 100, 1, 1000, 10000]
for lam in lambdas:
    for sigma in sigmas:
        print('---------------------------------------- Training with lambda %s and sigma %s ----------------------------------------' %(str(lam), str(sigma)))
        with open(os.path.join(regionClassifier.output_folder, "result.txt"), "a") as fid:
            fid.write('---------------------------------------- Training with lambda %s and sigma %s ----------------------------------------\n' %(str(lam), str(sigma)))
        regionClassifier.lam = lam
        regionClassifier.sigma = sigma
        # - Train region classifier
        model = regionClassifier.trainRegionClassifier()
            
        #quit()
        #model = regionClassifier.trainRegionClassifier(opts={'is_rpn': True, 'lam': lam, 'sigma': sigma})
        #torch.save(model, os.path.join(regionClassifier.output_folder, 'model_classifier_ep8_lambda%s_sigma%s_test' %(str(lam).replace(".","_"), str(sigma).replace(".","_"))))
        #model = torch.load('/home/IIT.LOCAL/fceola/workspace/ws_mask/python-online-detection/experiments/first_experiment/cls_detector')

        # Train region refiner
        region_refiner = RegionRefiner(cfg_online_path)
        #region_refiner = RegionRefiner('first_experiment/configs/config_region_refiner_server.yaml')
        COXY['X'] = COXY['X'] - stats['mean']
        COXY['X'] = COXY['X'] * (20 / stats['mean_norm'].item())
        region_refiner.COXY = COXY
        lam = 1000
        print('----------------------------------------------------------- Training with lambda %s' %(str(lam).replace(".","_")), '-----------------------------------------------------------')
        region_refiner.lambd = lam
        models = region_refiner.trainRegionRefiner()
        #torch.save(models, os.path.join(regionClassifier.output_folder, 'regressor_lambda%s_test' %(str(lam).replace(".","_"))))
        #torch.save(models,'first_experiment/cv_rpn_icwt21_online_pipeline/regressor_lambda%s_test' %(str(lam).replace(".","_")))


        # - Test region classifier (on validation set)
        print('Skip Test region classifier on validation set')

        # - Test region refiner (on validation set)
        print('Skip Test region refiner on validation set')

        # - Save/store results
        print('Skip saving model')

        # ----------------------------------------------------------------------------------
        # --------------------------------- Testing models ---------------------------------
        # ----------------------------------------------------------------------------------
        feature_extractor.is_train = False
        feature_extractor.is_test = True
        test_boxes = feature_extractor.extractFeatures()
        # Test the best classifier on the test set
        print('Region classifier test on the test set')
        predictions = regionClassifier.testRegionClassifier(model, test_boxes)
        #torch.save(predictions, '/home/IIT.LOCAL/fceola/workspace/ws_mask/python-online-detection/experiments/first_experiment/predictions')
        # region_refiner.boxes = predictions
        #predictions = torch.load('/home/IIT.LOCAL/fceola/workspace/ws_mask/python-online-detection/experiments/first_experiment/predictions')

        print('Region classifier predictions evaluation')
        result_cls = accuracy_evaluator.evaluate(dataset.dataset, copy.deepcopy(predictions), is_target_task=True,
                                                 cls_agnostic_bbox_reg=True, icwt_21_objs=icwt_21_objs)

        # Test the best regressor on the test set
        print('Region refiner test on the test set')
        region_refiner.boxes = predictions
        region_refiner.stats = stats
        region_refiner.feat = test_boxes
        refined_predictions = region_refiner.predict()

        print('Region classifier predictions evaluation')
        result_reg = accuracy_evaluator.evaluate(dataset.dataset, refined_predictions, is_target_task=True,
                                                 cls_agnostic_bbox_reg=False, icwt_21_objs=icwt_21_objs)
