import os
import sys

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'region-classifier')))
from maskrcnn_pytorch.benchmark.config import cfg

import OnlineRegionClassifier as ocr
import FALKONWrapper_with_centers_selection_logistic_loss as falkon
#import FALKONWrapper_with_centers_selection as falkon
#import FALKONWrapper as falkon
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
lambdas = [0.000001, 0.0000001, 0.0001, 0.00001, 0.001]
sigmas = [10, 15, 20, 25, 30, 50, 100, 1, 5, 1000, 10000]
for lam in lambdas:
    for sigma in sigmas:
        #if os.path.exists('first_experiment/cv_falkon_with_centers_selection_dataset_sampled_logistic_loss/model_classifier_rpn_ep5_lambda%s_sigma%s' %(str(lam).replace(".","_"), str(sigma).replace(".","_"))):
        #    continue
        models = regionClassifier.trainRegionClassifier(opts={'is_rpn': True, 'lam': lam, 'sigma': sigma})
        """
        mod = None
        for model in models:
            if model is not None:
                if mod is not None:
                    print('nystrom', torch.equal(model.ny_points_, mod.ny_points_))
                print(model == mod)
                mod = model
        """
        torch.save(models, 'first_experiment/cv_falkon_with_centers_selection_logistic_loss_new_easy_hard_neg_thresh/model_classifier_rpn_ep5_lambda%s_sigma%s' %(str(lam).replace(".","_"), str(sigma).replace(".","_")))
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


