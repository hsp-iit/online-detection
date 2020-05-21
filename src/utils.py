import math
import numpy as np
import os


def computeFeatStatistics(positives, negatives, num_samples=4000):
    print('To implement computeFeatStatistics in utils')
    pos_fraction = 1/10
    neg_fraction = 9/10
    num_classes = len(positives)
    take_from_pos = math.ceil((num_samples/num_classes)*pos_fraction)
    take_from_neg = math.ceil(((num_samples/num_classes)*neg_fraction)/len(negatives[0]))

    sampled_X = positives[0][0] #TO CHECK
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
    return mean, std, mean_norm


def zScores(feat, mean, mean_norm, target_norm=20):
    feat = feat - mean
    feat = np.multiply(feat, (target_norm / mean_norm))
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
