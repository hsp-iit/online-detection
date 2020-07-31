import os
import sys

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'region-classifier')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'region-refiner')))
from maskrcnn_pytorch.benchmark.config import cfg

import OnlineRegionClassifier as ocr
import FALKONWrapper_with_centers_selection_logistic_loss as falkon
#import FALKONWrapper_with_centers_selection as falkon
#import FALKONWrapper as falkon
import RPNMinibootstrapSelector as ms
import RPNPositivesSelector as ps
from region_refiner import RegionRefiner
import torch

# ----------------------------------------------------------------------------------------
# ------------------------------- Experiment configuration -------------------------------
# ----------------------------------------------------------------------------------------
#cfg_online_path = '/home/IIT.LOCAL/fceola/workspace/ws_mask/python-online-detection/experiments/Configs/config_federico_server_classifier.yaml'
cfg_online_path = '/home/IIT.LOCAL/fceola/workspace/ws_mask/python-online-detection/experiments/Configs/config_federico_server_classifier_icwt_21.yaml'
# Region Classifier initialization
classifier = falkon.FALKONWrapper(cfg_path=cfg_online_path)
negative_selector = ms.RPNMinibootstrapSelector(cfg_online_path)
positive_selector = ps.RPNPositivesSelector(cfg_online_path)
regionClassifier = ocr.OnlineRegionClassifier(classifier, positive_selector, negative_selector, cfg_path=cfg_online_path)

# -----------------------------------------------------------------------------------
# --------------------------------- Training models ---------------------------------
# -----------------------------------------------------------------------------------

# - Train region classifier
#lambdas = [0.000001, 0.0000001, 0.0001, 0.00001, 0.001]
#sigmas = [10, 15, 20, 25, 30, 50, 100, 1, 5, 1000, 10000]
#lambdas = [0.000001]
#sigmas = [15]

lambdas = [0.00001]
sigmas = [1000]
for lam in lambdas:
    for sigma in sigmas:
        #if os.path.exists('first_experiment/cv_falkon_with_centers_selection_dataset_sampled_logistic_loss/model_classifier_rpn_ep5_lambda%s_sigma%s' %(str(lam).replace(".","_"), str(sigma).replace(".","_"))):
        #    continue
        models = regionClassifier.trainRegionClassifier(opts={'is_rpn': True, 'lam': lam, 'sigma': sigma})
        #quit()
        torch.save(models, 'first_experiment/cv_rpn_icwt21_validation_set/model_classifier_rpn_ep8_lambda%s_sigma%s_test' %(str(lam).replace(".","_"), str(sigma).replace(".","_")))

region_refiner = RegionRefiner('first_experiment/configs/config_region_refiner_server_icwt_21.yaml')
#region_refiner = RegionRefiner('first_experiment/configs/config_region_refiner_server.yaml')

## Retrieve feature extractor (either by loading it or by training it)
#try:
#    feature_extractor.loadFeatureExtractor()
#except OSError:
#    print('Feature extractor will be trained from scratch.')
#    feature_extractor.trainFeatureExtractor()

## Extract features for the train/val/test sets
#feature_extractor.extractFeatures()

#lambdas = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
lambdas = [0.01]
for lam in lambdas:
    print('----------------------------------------------------------- Training with lambda %s' %(str(lam).replace(".","_")), '-----------------------------------------------------------')
    region_refiner.lambd = lam
    models = region_refiner.trainRegionRefiner()
    #torch.save(models,'first_experiment/cv_rpn_icwt30_validation_set/regressor_lambda%s' %(str(lam).replace(".","_")))
    torch.save(models,'first_experiment/cv_rpn_icwt21_validation_set/regressor_lambda%s_test' %(str(lam).replace(".","_")))

