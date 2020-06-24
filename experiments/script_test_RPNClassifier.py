import os
import sys

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'region-classifier')))
from maskrcnn_pytorch.benchmark.config import cfg

import OnlineRegionClassifier as ocr
import FALKONWrapper as falkon
import RPNMinibootstrapSelector as ms
import RPNPositivesSelector as ps

import torch

# ----------------------------------------------------------------------------------------
# ------------------------------- Experiment configuration -------------------------------
# ----------------------------------------------------------------------------------------
cfg_online_path = '/home/IIT.LOCAL/fceola/workspace/ws_mask/python-online-detection/experiments/Configs/config_federico_server_classifier.yaml'

# Region Classifier initialization
classifier = falkon.FALKONWrapper(cfg_path=cfg_online_path)
negative_selector = ms.RPNMinibootstrapSelector(cfg_online_path)
positive_selector = ps.RPNPositivesSelector(cfg_online_path)
regionClassifier = ocr.OnlineRegionClassifier(classifier, positive_selector, negative_selector, cfg_path=cfg_online_path)

# -----------------------------------------------------------------------------------
# --------------------------------- Training models ---------------------------------
# -----------------------------------------------------------------------------------

# - Train region classifier
models = regionClassifier.trainRegionClassifier(opts={'is_rpn': True})
mod = None
for model in models:
    if model is not None:
        if mod is not None:
            print('nystrom', torch.equal(model.ny_points_, mod.ny_points_))
        print(model == mod)
        mod = model
torch.save(models, 'model_classifier_rpn_ep5_lmd0_0001_sigma5')
"""
for model in models:
    torch.save(model, 'model')
    print('saved')
models_names = ['model'+str(i) for i in range(len(models))]
models_dictionary = {k:v for k, v in zip(models_names, models)}


#sys.setrecursionlimit(50000)
#torch.save(model.state_dict(), 'model_classifier_rpn_')
torch.save(models_dictionary, 'model_classifier_rpn_')

import shelve


my_shelf = shelve.open('rpn_classifier_model','n') # 'n' for new

my_shelf['model'] = model
my_shelf.close()
"""


