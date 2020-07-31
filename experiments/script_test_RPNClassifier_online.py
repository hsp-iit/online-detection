import os
import sys

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'region-classifier')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'region-refiner')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'feature-extractor')))

from feature_extractor import FeatureExtractor
from maskrcnn_pytorch.benchmark.config import cfg

import OnlineRegionClassifierRPNOnline as ocr
import FALKONWrapper_with_centers_selection_logistic_loss as falkon
#import FALKONWrapper_with_centers_selection as falkon
#import FALKONWrapper as falkon
#import RPNMinibootstrapSelector as ms
#import RPNPositivesSelector as ps
from region_refiner import RegionRefiner
import torch
import math


feature_extractor = FeatureExtractor('first_experiment/configs/config_feature_task_federico.yaml', 'first_experiment/configs/config_target_task_FALKON_federico.yaml', 'first_experiment/configs/config_rpn_federico_icwt_21.yaml') # 'first_experiment/configs/config_rpn_federico.yaml') #

## Retrieve feature extractor (either by loading it or by training it)
try:
    feature_extractor.loadFeatureExtractor()
except OSError:
    print('Feature extractor will be trained from scratch.')
    #feature_extractor.trainFeatureExtractor()

negatives, positives, COXY = feature_extractor.extractRPNFeatures()
#quit()

def computeFeatStatistics(positives, negatives, num_samples=4000):
    print('Computing features statistics')
    pos_fraction = 1/10
    neg_fraction = 9/10
    num_classes = len(positives)
    take_from_pos = math.ceil((num_samples/num_classes)*pos_fraction)
    take_from_neg = math.ceil(((num_samples/num_classes)*neg_fraction)/len(negatives[0]))

    sampled_X = torch.empty((0,1024), device='cuda')
    ns = torch.empty((0,1), device='cuda')
    for i in range(num_classes):
        if len(positives[i]) != 0:
            #print(1)
            sampled_X = positives[i][0].unsqueeze(0)
            #try:
            ns = torch.cat((ns, torch.norm(positives[i][0].view(-1, 1024) , dim=1).view(-1,1)), dim=0)                             #ns = np.vstack((ns, np.transpose(np.linalg.norm(pos_picked.cpu(), axis=1)[np.newaxis])))
            #except:
            #    ns = torch.cat((ns, torch.norm(positives[i][0])), dim=0)                             #ns = np.vstack((ns, np.transpose(np.linalg.norm(pos_picked.cpu(), axis=1)[np.newaxis])))
            #print(ns)
    for i in range(num_classes):
        if len(positives[i]) != 0:
            #print(2)
            pos_idx = torch.randint(len(positives[i]), (take_from_pos,)) #np.random.randint(len(positives[i]), size=take_from_pos)
            pos_picked = positives[i][pos_idx]
            #print(sampled_X, pos_picked)
            sampled_X = torch.cat((sampled_X, pos_picked))      #np.vstack((sampled_X, pos_picked.cpu()))
            #ns = torch.cat((ns, torch.t(torch.norm(pos_picked))), dim=0)                             #ns = np.vstack((ns, np.transpose(np.linalg.norm(pos_picked.cpu(), axis=1)[np.newaxis])))
            #try:
            #ns = torch.cat((ns, torch.t(torch.norm(pos_picked))), dim=0)                             #ns = np.vstack((ns, np.transpose(np.linalg.norm(pos_picked.cpu(), axis=1)[np.newaxis])))
            ns = torch.cat((ns, torch.norm(pos_picked.view(-1, 1024) , dim=1).view(-1,1)), dim=0)
            #except:
            #    ns = torch.cat((ns, torch.norm(pos_picked).unsqueeze(0)), dim=0)                             #ns = np.vstack((ns, np.transpose(np.linalg.norm(pos_picked.cpu(), axis=1)[np.newaxis])))
        for j in range(len(negatives[i])):
            if len(negatives[i][j]) != 0:
                #print(3)
                """
                neg_idx = np.random.choice(len(negatives[i][j]), size=take_from_neg)
                neg_picked = negatives[i][j][neg_idx]
                #print(neg_picked.shape, sampled_X.shape)
                sampled_X = np.vstack((sampled_X, neg_picked.cpu()))
                ns = np.vstack((ns, np.transpose(np.linalg.norm(neg_picked.cpu(), axis=1)[np.newaxis])))
                """
                neg_idx = torch.randint(len(negatives[i][j]), (take_from_neg,)) #np.random.randint(len(positives[i]), size=take_from_pos)
                neg_picked = negatives[i][j][neg_idx]
                #print(sampled_X, neg_picked, i, j)
                sampled_X = torch.cat((sampled_X, neg_picked))      #np.vstack((sampled_X, pos_picked.cpu()))
                #try:
                #ns = torch.cat((ns, torch.t(torch.norm(neg_picked, dim=1))), dim=0)                             #ns = np.vstack((ns, np.transpose(np.linalg.norm(pos_picked.cpu(), axis=1)[np.newaxis])))
                ns = torch.cat((ns, torch.norm(neg_picked.view(-1, 1024) , dim=1).view(-1,1)), dim=0)
                #except:
                #    ns = torch.cat((ns, torch.norm(neg_picked)), dim=0)                             #ns = np.vstack((ns, np.transpose(np.linalg.norm(pos_picked.cpu(), axis=1)[np.newaxis])))

    mean = torch.mean(sampled_X, dim=0)
    std = torch.std(sampled_X, dim=0)
    mean_norm = torch.mean(ns)
    # print('Statistics computed. Mean: {}, Std: {}, Mean Norm {}'.format(mean.item(), std.item(), mean_norm.item()))
    stats = {'mean': mean, 'std': std, 'mean_norm': mean_norm}


    return stats
stats = computeFeatStatistics(positives, negatives)
#print(stats)
torch.save(stats,'first_experiment/cv_rpn_icwt21_online_pipeline/stats')
#torch.save(stats,'first_experiment/cv_rpn_icwt30_online_pipeline/stats')

#print(negatives, positives, stats)
#quit()
# ----------------------------------------------------------------------------------------
# ------------------------------- Experiment configuration -------------------------------
# ----------------------------------------------------------------------------------------
#cfg_online_path = '/home/IIT.LOCAL/fceola/workspace/ws_mask/python-online-detection/experiments/Configs/config_federico_server_classifier.yaml'
cfg_online_path = '/home/IIT.LOCAL/fceola/workspace/ws_mask/python-online-detection/experiments/Configs/config_federico_server_classifier_icwt_21.yaml'
# Region Classifier initialization
classifier = falkon.FALKONWrapper(cfg_path=cfg_online_path)
#negative_selector = ms.RPNMinibootstrapSelector(cfg_online_path)
#positive_selector = ps.RPNPositivesSelector(cfg_online_path)
regionClassifier = ocr.OnlineRegionClassifier(classifier, positives, negatives, stats, cfg_path=cfg_online_path)

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
#lambdas = [0.001]
#sigmas = [20]
for lam in lambdas:
    for sigma in sigmas:
        #if os.path.exists('first_experiment/cv_falkon_with_centers_selection_dataset_sampled_logistic_loss/model_classifier_rpn_ep5_lambda%s_sigma%s' %(str(lam).replace(".","_"), str(sigma).replace(".","_"))):
        #    continue
        models = regionClassifier.trainRegionClassifier(opts={'is_rpn': True, 'lam': lam, 'sigma': sigma})
        #quit()
        torch.save(models, 'first_experiment/cv_rpn_icwt21_online_pipeline/model_classifier_rpn_ep8_lambda%s_sigma%s_test' %(str(lam).replace(".","_"), str(sigma).replace(".","_")))
        #torch.save(models, 'first_experiment/cv_rpn_icwt30_online_pipeline/model_classifier_rpn_ep5_lambda%s_sigma%s_test' %(str(lam).replace(".","_"), str(sigma).replace(".","_")))

#quit()

region_refiner = RegionRefiner('first_experiment/configs/config_region_refiner_server_icwt_21.yaml')
#region_refiner = RegionRefiner('first_experiment/configs/config_region_refiner_server.yaml')
COXY['X'] = COXY['X'] - stats['mean']
COXY['X'] = COXY['X'] * (20 / stats['mean_norm'].item())
region_refiner.COXY = COXY
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
    #torch.save(models,'first_experiment/cv_rpn_icwt30_online_pipeline/regressor_lambda%s_test' %(str(lam).replace(".","_")))
    torch.save(models,'first_experiment/cv_rpn_icwt21_online_pipeline/regressor_lambda%s_test' %(str(lam).replace(".","_")))

