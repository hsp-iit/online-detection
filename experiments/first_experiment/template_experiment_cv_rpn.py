import os
import sys


basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..')))
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..', 'src', 'modules', 'feature-extractor')))
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..', 'src', 'modules', 'region-classifier')))
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..', 'src', 'modules', 'region-refiner')))

from feature_extractor import FeatureExtractor
#import region_classifier
#import region_refiner
from shutil import copyfile
## Experiment configuration
feature_extractor = FeatureExtractor('configs/config_feature_task_federico.yaml', 'configs/config_target_task_FALKON_federico.yaml')
"""
## Retrieve feature extractor (either by loading it or by training it)
try:
    feature_extractor.loadFeatureExtractor()
except OSError:
    print('Feature extractor will be trained from scratch.')
    feature_extractor.trainFeatureExtractor()
"""
## Extract features for the train/val/test sets
lambdas = [0.000001, 0.0000001, 0.0001, 0.00001, 0.001]
sigmas = [10, 15, 20, 25, 30, 50, 100, 1, 5, 1000, 10000]
for lam in lambdas:
    for sigma in sigmas:
        print('---------------------------------------- Computing average recall with lambda %s and sigma %s ----------------------------------------' %(str(lam), str(sigma)))
        copyfile('cv_classifier_falkon_m1000_original_easy_hard_thresh/model_classifier_rpn_ep5_lambda%s_sigma%s' %(str(lam).replace(".","_"), str(sigma).replace(".","_")), 'cv_falkon_obj_1000_or_thresh')
        feature_extractor.extractFeatures()

## Train region refiner

## Start the cross validation

### - Set parameters
### - Train region classifier
### - Test region classifier (on validation set)
### - Test region refiner (on validation set)
### - Save/store results

## Test the best model (on the test set)
