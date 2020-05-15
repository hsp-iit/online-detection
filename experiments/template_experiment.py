import os
import sys

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, '..', 'src', 'modules', 'feature-extractor')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, '..', 'src', 'modules', 'region-classifier')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, '..', 'src', 'modules', 'region-refiner')))

import feature_extractor
import region_classifier
import region_refiner

## Experiment configuration

## Retrieve feature extractor (either by loading it or by training it)

## Extract features for the train/val/test sets

## Train region refiner

## Start the cross validation

### - Set parameters
### - Train region classifier
### - Test region classifier (on validation set)
### - Test region refiner (on validation set)
### - Save/store results

## Test the best model (on the test set)