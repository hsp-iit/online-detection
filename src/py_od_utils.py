import math
import numpy as np
import os
import torch


def computeFeatStatistics(positives, negatives, feature_folder, is_rpn, num_samples=4000,):
    basedir = os.path.dirname(__file__)
    if not is_rpn:
        stats_path = os.path.join(basedir, '..', 'Data', 'feat_cache', feature_folder, 'stats')
    else:
        stats_path = os.path.join(basedir, '..', 'Data', 'feat_cache_RPN', feature_folder, 'rpn_stats')
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

        sampled_X = 0
        ns = 0
        for i in range(num_classes):
            if len(positives[i]) != 0:
                sampled_X = positives[i][0].cpu()
                ns = np.transpose(np.linalg.norm(positives[i][0].cpu()))
        for i in range(num_classes):
            if len(positives[i]) != 0:
                pos_idx = np.random.randint(len(positives[i]), size=take_from_pos)
                pos_picked = positives[i][pos_idx]
                sampled_X = np.vstack((sampled_X, pos_picked.cpu()))
                ns = np.vstack((ns, np.transpose(np.linalg.norm(pos_picked.cpu(), axis=1)[np.newaxis])))
            for j in range(len(negatives[i])):
                if len(negatives[i][j]) != 0:
                    neg_idx = np.random.choice(len(negatives[i][j]), size=take_from_neg)
                    neg_picked = negatives[i][j][neg_idx]
                    #print(neg_picked.shape, sampled_X.shape)
                    sampled_X = np.vstack((sampled_X, neg_picked.cpu()))
                    ns = np.vstack((ns, np.transpose(np.linalg.norm(neg_picked.cpu(), axis=1)[np.newaxis])))

        mean = np.mean(sampled_X, axis=0)
        std = np.std(sampled_X, axis=0)
        mean_norm = np.mean(ns)

        mean = torch.tensor(mean)
        std = torch.tensor(std)
        mean_norm = torch.tensor(mean_norm)

        # print('Statistics computed. Mean: {}, Std: {}, Mean Norm {}'.format(mean.item(), std.item(), mean_norm.item()))
        l = {'mean': mean, 'std': std, 'mean_norm': mean_norm}
        #print(l)
        #quit()
        torch.save(l, stats_path)

    return mean, std, mean_norm


def computeFeatStatistics_torch(positives, negatives, num_samples=4000, features_dim=2048):
    print('Computing features statistics')
    pos_fraction = 1/10
    neg_fraction = 9/10
    num_classes = len(positives)
    take_from_pos = math.ceil((num_samples/num_classes)*pos_fraction)
    take_from_neg = math.ceil(((num_samples/num_classes)*neg_fraction)/len(negatives[0]))
    sampled_X = torch.empty((0, features_dim), device='cuda')
    ns = torch.empty((0,1), device='cuda')
    for i in range(num_classes):
        if len(positives[i]) != 0:
            sampled_X = positives[i][0].unsqueeze(0)
            ns = torch.cat((ns, torch.norm(positives[i][0].view(-1, features_dim) , dim=1).view(-1,1)), dim=0)
    for i in range(num_classes):
        if len(positives[i]) != 0:
            pos_idx = torch.randint(len(positives[i]), (take_from_pos,))
            pos_picked = positives[i][pos_idx]
            sampled_X = torch.cat((sampled_X, pos_picked))
            ns = torch.cat((ns, torch.norm(pos_picked.view(-1, features_dim) , dim=1).view(-1,1)), dim=0)
        for j in range(len(negatives[i])):
            if len(negatives[i][j]) != 0:
                neg_idx = torch.randint(len(negatives[i][j]), (take_from_neg,))
                neg_picked = negatives[i][j][neg_idx]
                sampled_X = torch.cat((sampled_X, neg_picked))
                ns = torch.cat((ns, torch.norm(neg_picked.view(-1, features_dim) , dim=1).view(-1,1)), dim=0)

    mean = torch.mean(sampled_X, dim=0)
    std = torch.std(sampled_X, dim=0)
    mean_norm = torch.mean(ns)
    stats = {'mean': mean, 'std': std, 'mean_norm': mean_norm}

    return stats


def zScores(feat, mean, mean_norm, target_norm=20):
    feat = torch.tensor(feat)
    feat = feat - mean
    feat = feat * (target_norm / mean_norm)
    return feat


def normalize_COXY(COXY, stats):
    COXY['X'] = COXY['X'] - stats['mean']
    COXY['X'] = COXY['X'] * (20 / stats['mean_norm'].item())
    return COXY

def falkon_models_to_cuda(models):
    for i in range(len(models)):
        if models[i] is not None:
            models[i].ny_points_ = models[i].ny_points_.to('cuda')
            models[i].alpha_ = models[i].alpha_.to('cuda')
    return models


