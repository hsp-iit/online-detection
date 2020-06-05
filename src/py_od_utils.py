import math
import numpy as np
import os
import torch


def getFeatPath(cfg):
    s =  '' + cfg['FEATURE_INFO']['BACKBONE'] + '_ep' \
            + str(cfg['FEATURE_INFO']['NUM_EPOCHS']) + '_FT' \
            + cfg['FEATURE_INFO']['FEAT_TASK_NAME'] + '_TT' \
            + cfg['FEATURE_INFO']['TARGET_TASK_NAME'] + ''
    return s


def computeFeatStatistics(positives, negatives, feature_folder, num_samples=4000):
    basedir = os.path.dirname(__file__)
    stats_path = os.path.join(basedir, '..', 'Data', 'feat_cache', feature_folder, 'stats')
    try:
        l = torch.load(stats_path)
        mean = torch.tensor(l['mean'])
        std = torch.tensor(l['std'])
        mean_norm = torch.tensor(l['mean_norm'])
    except:
        print('Computing features statistics')
        pos_fraction = 1/10
        neg_fraction = 9/10
        num_classes = len(positives)
        take_from_pos = math.ceil((num_samples/num_classes)*pos_fraction)
        take_from_neg = math.ceil(((num_samples/num_classes)*neg_fraction)/len(negatives[0]))

        sampled_X = positives[0][0]
        ns = np.transpose(np.linalg.norm(positives[0][0]))
        for i in range(num_classes):
            pos_idx = np.random.randint(len(positives[i]), size=take_from_pos)
            pos_picked = positives[i][pos_idx]
            sampled_X = np.vstack((sampled_X, pos_picked))
            ns = np.vstack((ns, np.transpose(np.linalg.norm(pos_picked, axis=1)[np.newaxis])))
            for j in range(len(negatives[i])):
                neg_idx = np.random.choice(len(negatives[i][j]), size=take_from_neg)
                neg_picked = negatives[i][j][neg_idx]
                sampled_X = np.vstack((sampled_X, neg_picked))
                ns = np.vstack((ns, np.transpose(np.linalg.norm(neg_picked, axis=1)[np.newaxis])))

        mean = np.mean(sampled_X, axis=0)
        std = np.std(sampled_X, axis=0)
        mean_norm = np.mean(ns)

        mean = torch.tensor(mean)
        std = torch.tensor(std)
        mean_norm = torch.tensor(mean_norm)

        print('Statistics computed. Mean: {}, Std: {}, Mean Norm {}'.format(mean.item(), std.item(), mean_norm.item()))
        l = {'mean': mean, 'std': std, 'mean_norm': mean_norm}
        torch.save(l, stats_path)

    return mean, std, mean_norm


def zScores(feat, mean, mean_norm, target_norm=20):
    feat = torch.tensor(feat)
    feat = feat - mean
    feat = feat * (target_norm / mean_norm)
    return feat


def loadFeature(feat_path, file_name, type='mat'):
    file_path_noExt = os.path.join(feat_path, file_name)
    if type == 'mat':
        import scipy.io
        file_path = file_path_noExt + '.mat'
        feature = scipy.io.loadmat(file_path)
    else:
        print('Unrecognized feature type: {}'.format(type))
        feature = None

    return feature
