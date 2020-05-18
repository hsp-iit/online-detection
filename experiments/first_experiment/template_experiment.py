import os
import sys

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..', 'src', 'modules', 'feature-extractor')))
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..', 'src', 'modules', 'region-classifier')))
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..', 'src', 'modules', 'region-refiner')))

from feature_extractor import FeatureExtractor
#import region_classifier
#import region_refiner

## Experiment configuration
a = FeatureExtractor('configs/config_feature_task.yaml', 'configs/config_target_task_FALKON.yaml')

## Retrieve feature extractor (either by loading it or by training it)
a.loadFeatureExtractor()
a.trainFeatureExtractor()

## Extract features for the train/val/test sets
a.extractFeatures()

## Train region refiner

## Start the cross validation

### - Set parameters
### - Train region classifier
### - Test region classifier (on validation set)
### - Test region refiner (on validation set)
### - Save/store results

## Test the best model (on the test set)