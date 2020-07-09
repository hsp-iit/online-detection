import os
import sys
import torch


basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..')))
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..', 'src', 'modules', 'feature-extractor')))
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..', 'src', 'modules', 'region-classifier')))
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..', 'src', 'modules', 'region-refiner')))

from feature_extractor import FeatureExtractor
#import region_classifier
from region_refiner import RegionRefiner

## Experiment configuration
#feature_extractor = FeatureExtractor('configs/config_feature_task.yaml', 'configs/config_target_task_FALKON.yaml')
region_refiner = RegionRefiner('configs/config_region_refiner_server.yaml')

## Retrieve feature extractor (either by loading it or by training it)
#try:
#    feature_extractor.loadFeatureExtractor()
#except OSError:
#    print('Feature extractor will be trained from scratch.')
#    feature_extractor.trainFeatureExtractor()

## Extract features for the train/val/test sets
#feature_extractor.extractFeatures()
"""
## Train region refiner
lambdas = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01]
#lambdas = [0.01, 0.1, 1, 10, 100, 1000]
sigmas = [10, 15, 20, 25, 30, 50, 100, 1, 5, 1000, 10000]
for lam in lambdas:
    for sigma in sigmas:
        print('----------------------------------------------------------- Training with lambda %f and sigma %i' %(lam, sigma), '-----------------------------------------------------------')
        #if os.path.exists('first_experiment/cv_falkon_with_centers_selection_dataset_sampled_logistic_loss/model_classifier_rpn_ep5_lambda%s_sigma%s' %(str(lam).replace(".","_"), str(sigma).replace(".","_"))):
        #    continue
        region_refiner.sigma = sigma
        region_refiner.lambd = lam
        models = region_refiner.trainRegionRefiner()
        
        #mod = None
        #for model in models:
        #    if model is not None:
        #        if mod is not None:
        #            print('nystrom', torch.equal(model.ny_points_, mod.ny_points_))
        #        print(model == mod)
        #        mod = model
        
        torch.save(models, 'cv_regressors_falkon_m30000_train_with_test_set/cv_lambda%s_sigma%s' %(str(lam).replace(".","_"), str(sigma).replace(".","_")))
"""
#lambdas = [0.0000000001, 0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
lambdas = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
for lam in lambdas:
    print('----------------------------------------------------------- Training with lambda %s' %(str(lam).replace(".","_")), '-----------------------------------------------------------')
    region_refiner.lambd = lam
    models = region_refiner.trainRegionRefiner()
    torch.save(models,'cv_regressors_linear/cv_lambda%s' %(str(lam).replace(".","_")))

## Start the cross validation
quit()
### - Set parameters
### - Train region classifier
### - Test region classifier (on validation set)
### - Test region refiner (on validation set)
refined_regions = region_refiner.predict()
### - Save/store results
torch.save(refined_regions, 'test_regressors_mask')
## Test the best model (on the test set)
