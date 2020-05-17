import os
import sys

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'region-classifier')))

import OnlineRegionClassifier as ocr
import FALKONWrapper as falkon
import MinibootstrapSelector as ms

# Experiment configuration
experiment_name = 'test_classifier'
classifier = falkon.FALKONWrapper()
negative_selector = ms.MinibootstrapSelector(10, 2000)
regionClassifier = ocr.OnlineRegionClassifier(experiment_name, classifier, negative_selector)
opts = dict()
opts['kernel_type'] = 'gauss'
opts['num_classes'] = 30

# Retrieve feature extractor (either by loading it or by training it)
print('Skip retriever feature extractor')

# Extract features for the train/val/test sets
print('Skip feature extraction')
dataset = None

# Train region refiner
print('Skip region refiner training')

# Start the cross validation
print('Skip cross validation')

# - Set parameters
opts['lambda'] = 0.001
opts['sigma'] = 15
opts['M'] = 2000

# - Train region classifier
model = regionClassifier.trainRegionClassifier(dataset, opts)

# - Test region classifier (on validation set)
print('Skip Test region classifier on validation set')

# - Test region refiner (on validation set)
print('Skip Test region refiner on validation set')

# - Save/store results
print('Skip saving model')

# Test the best model (on the test set)
results = regionClassifier.testRegionClassifier(dataset)

# Test region refiner (on test set)
print('Skip Test region refiner on test set')

