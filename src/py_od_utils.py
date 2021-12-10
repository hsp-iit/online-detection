import math
import numpy as np
import os
import torch
import glob
import yaml

def computeFeatStatistics(positives, negatives, feature_folder, is_rpn, num_samples=4000):
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
                    sampled_X = np.vstack((sampled_X, neg_picked.cpu()))
                    ns = np.vstack((ns, np.transpose(np.linalg.norm(neg_picked.cpu(), axis=1)[np.newaxis])))

        mean = np.mean(sampled_X, axis=0)
        std = np.std(sampled_X, axis=0)
        mean_norm = np.mean(ns)

        mean = torch.tensor(mean)
        std = torch.tensor(std)
        mean_norm = torch.tensor(mean_norm)
        l = {'mean': mean, 'std': std, 'mean_norm': mean_norm}
        torch.save(l, stats_path)

    return mean, std, mean_norm


def computeFeatStatistics_torch(positives, negatives, num_samples=4000, features_dim=2048, cpu_tensor=False, pos_fraction=None):
    device = 'cpu' if cpu_tensor else 'cuda'
    print('Computing features statistics')
    if pos_fraction is None:
        pos_fraction = 1/10
        neg_fraction = 9/10
    else:
        neg_fraction = 1 - pos_fraction
    num_classes = len(positives)
    take_from_pos = math.ceil((num_samples/num_classes)*pos_fraction)
    nb = 0
    for i in range(len(negatives)):
        if len(negatives[i]) > nb:
            nb = len(negatives[i])
    #take_from_neg = math.ceil(((num_samples/num_classes)*neg_fraction)/len(negatives[0]))
    take_from_neg = math.ceil(((num_samples / num_classes) * neg_fraction) / nb)
    sampled_X = torch.empty((0, features_dim), device=device)
    ns = torch.empty((0,1), device=device)
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
    stats = {'mean': mean.to('cuda'), 'std': std.to('cuda'), 'mean_norm': mean_norm.to('cuda')}

    return stats


def zScores(feat, mean, mean_norm, target_norm=20):
    feat = torch.tensor(feat)
    feat = feat - mean
    feat = feat * (target_norm / mean_norm)
    return feat


def normalize_COXY(COXY, stats, cpu=False):
    if cpu:
        COXY['X'] = COXY['X'] - stats['mean'].to('cpu')
    else:
        COXY['X'] = COXY['X'] - stats['mean']
    COXY['X'] = COXY['X'] * (20 / stats['mean_norm'].item())
    return COXY

def falkon_models_to_cuda(models):
    for i in range(len(models)):
        if models[i] is not None:
            models[i].ny_points_ = models[i].ny_points_.to('cuda')
            models[i].alpha_ = models[i].alpha_.to('cuda')
    return models

def load_features_classifier(features_dir, is_segm=False, cpu_tensor=False, sample_ratio=1, cfg_feature_extraction=None):
    positives_to_load = len(glob.glob(os.path.join(features_dir, 'positives_*')))
    positives_loaded = 0
    negatives_to_load = len(glob.glob(os.path.join(features_dir, 'negatives_*')))
    negatives_loaded = 0
    positives = []
    negatives = []
    clss_id = 0
    shuffle_features = False
    # If features must be shuffled, the number of batches and the batch size must be specified
    # Default parameters for the feature extraction cfg files
    bs_shuffled = 2000
    nb_shuffled = 2
    if cfg_feature_extraction is not None:
        feat_extraction_params_cfg_file = open(cfg_feature_extraction)
        feat_extraction_params = yaml.load(feat_extraction_params_cfg_file, Loader=yaml.FullLoader)
        if 'MINIBOOTSTRAP' in feat_extraction_params:
            if 'RPN' in feat_extraction_params['MINIBOOTSTRAP'] and 'RPN' in features_dir:
                if 'SHUFFLE_NEGATIVES' in feat_extraction_params['MINIBOOTSTRAP']['RPN']:
                    shuffle_features = feat_extraction_params['MINIBOOTSTRAP']['RPN']['SHUFFLE_NEGATIVES']
                if 'ITERATIONS' in feat_extraction_params['MINIBOOTSTRAP']['RPN']:
                    nb_shuffled = feat_extraction_params['MINIBOOTSTRAP']['RPN']['ITERATIONS']
                if 'BATCH_SIZE' in feat_extraction_params['MINIBOOTSTRAP']['RPN']:
                    bs_shuffled = feat_extraction_params['MINIBOOTSTRAP']['RPN']['BATCH_SIZE']
            if 'DETECTOR' in feat_extraction_params['MINIBOOTSTRAP'] and 'detector' in features_dir:
                if 'SHUFFLE_NEGATIVES' in feat_extraction_params['MINIBOOTSTRAP']['DETECTOR']:
                    shuffle_features = feat_extraction_params['MINIBOOTSTRAP']['DETECTOR']['SHUFFLE_NEGATIVES']
                if 'ITERATIONS' in feat_extraction_params['MINIBOOTSTRAP']['DETECTOR']:
                    nb_shuffled = feat_extraction_params['MINIBOOTSTRAP']['DETECTOR']['ITERATIONS']
                if 'BATCH_SIZE' in feat_extraction_params['MINIBOOTSTRAP']['DETECTOR']:
                    bs_shuffled = feat_extraction_params['MINIBOOTSTRAP']['DETECTOR']['BATCH_SIZE']
    while positives_loaded < positives_to_load or negatives_loaded < negatives_to_load:
        # Load positives with class id clss_id
        positives_to_load_i = len(glob.glob(os.path.join(features_dir, 'positives_cl_{}_*'.format(clss_id))))
        positives_i = []
        for batch in range(positives_to_load_i):
            positives_i.append(torch.load(os.path.join(features_dir,'positives_cl_{}_batch_{}'.format(clss_id, batch))))
            positives_loaded += 1
        # If there are not positives for this class, add an empty tensor
        try:
            if not cpu_tensor:
                to_append_pos_i = torch.cat(positives_i)
                if sample_ratio < 1:
                    indices = torch.randint(len(to_append_pos_i), (int(len(to_append_pos_i)*sample_ratio),))
                    to_append_pos_i = to_append_pos_i[indices]
                positives.append(to_append_pos_i)
            else:
                positives.append(torch.cat(positives_i).to('cpu'))
        except:
            positives.append(torch.empty((0)))
        if is_segm:
            negatives_to_load_i = len(glob.glob(os.path.join(features_dir, 'negatives_cl_{}_*'.format(clss_id))))
            negatives_i = []
            for batch in range(negatives_to_load_i):
                negatives_i.append(torch.load(os.path.join(features_dir, 'negatives_cl_{}_batch_{}'.format(clss_id, batch))))
                negatives_loaded += 1
            # If there are not negatives for this class, add an empty tensor
            try:
                if not cpu_tensor:
                    to_append_neg_i = torch.cat(negatives_i)
                    if sample_ratio < 1:
                        indices = torch.randint(len(to_append_neg_i), (int(len(to_append_neg_i) * sample_ratio),))
                        to_append_neg_i = to_append_neg_i[indices]
                    negatives.append(to_append_neg_i)
                else:
                    negatives.append(torch.cat(negatives_i).to('cpu'))
            except:
                negatives.append(torch.empty((0)))
        else:
            # Load negatives with class id clss_id
            negatives_to_load_i = len(glob.glob(os.path.join(features_dir, 'negatives_cl_{}_*'.format(clss_id))))
            negatives_i = []
            for batch in range(negatives_to_load_i):
                negatives_i.append(torch.load(os.path.join(features_dir,'negatives_cl_{}_batch_{}'.format(clss_id, batch))))
                negatives_loaded += 1
            negatives.append(negatives_i)
        clss_id += 1
    if not is_segm and shuffle_features:
        negatives = shuffle_negatives(negatives, batch_size=bs_shuffled, num_batches=nb_shuffled)

    return positives, negatives

def load_features_regressor(features_dir, samples_fraction=1.0):
    reg_num_batches = len(glob.glob(os.path.join(features_dir, 'reg_x_*')))
    X_list = []
    C_list = []
    Y_list = []
    for i in range(reg_num_batches):
        if samples_fraction < 1.0:
            C_i = torch.load(os.path.join(features_dir, 'reg_c_batch_{}'.format(i)))
            ind_i = torch.randperm(len(C_i))[:int(len(C_i)*samples_fraction)]
            X_list.append(torch.load(os.path.join(features_dir, 'reg_x_batch_{}'.format(i)))[ind_i])
            C_list.append(C_i[ind_i])
            Y_list.append(torch.load(os.path.join(features_dir, 'reg_y_batch_{}'.format(i)))[ind_i])
        else:
            X_list.append(torch.load(os.path.join(features_dir, 'reg_x_batch_{}'.format(i))))
            C_list.append(torch.load(os.path.join(features_dir, 'reg_c_batch_{}'.format(i))))
            Y_list.append(torch.load(os.path.join(features_dir, 'reg_y_batch_{}'.format(i))))

    COXY = {'C': torch.cat(C_list),
            'O': None,
            'X': torch.cat(X_list),
            'Y': torch.cat(Y_list)
            }
    return COXY

def load_positives_from_COXY(COXY, del_COXY=False, samples_fraction=1.0):
    positives = []
    num_classes = len(torch.unique(COXY['C']))      #TODO, this should be modified, if there are not positives for one class it can cause errors
    for i in range(num_classes):
        ids_i = torch.where(COXY['C'] == i+1)[0]
        if samples_fraction < 1.0:
            ids_i = ids_i[torch.randperm(len(ids_i))[:int(len(ids_i)*samples_fraction)]]
        positives.append(COXY['X'][ids_i])
        if del_COXY:
            ids_i_to_rm = torch.where(COXY['C'] != i + 1)[0]
            COXY['X'] = COXY['X'][ids_i_to_rm]
            COXY['C'] = COXY['C'][ids_i_to_rm]

    return positives

def minibatch_positives(positives, num_batches):
    for i in range(len(positives)):
        positives_per_batch = int(len(positives[i])/num_batches)
        positives[i] = list(torch.split(positives[i], positives_per_batch))
    return positives

def decode_boxes_detector(boxes, bbox_pred):
    ex_box = boxes.bbox

    dst_ctr_x = bbox_pred[:, 0::4]
    dst_ctr_y = bbox_pred[:, 1::4]
    dst_scl_x = bbox_pred[:, 2::4]
    dst_scl_y = bbox_pred[:, 3::4]

    src_w = ex_box[:, 2] - ex_box[:, 0] + 1
    src_h = ex_box[:, 3] - ex_box[:, 1] + 1
    src_ctr_x = ex_box[:, 0] + 0.5 * src_w
    src_ctr_y = ex_box[:, 1] + 0.5 * src_h
    pred_ctr_x = (dst_ctr_x * src_w[:, None]) + src_ctr_x[:, None]
    pred_ctr_y = (dst_ctr_y * src_h[:, None]) + src_ctr_y[:, None]
    pred_w = torch.exp(dst_scl_x) * src_w[:, None]
    pred_h = torch.exp(dst_scl_y) * src_h[:, None]
    pred_boxes = torch.zeros_like(bbox_pred)
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

    pred_boxes[:, 0::4] = torch.max(pred_boxes[:, 0::4], torch.zeros(pred_boxes[:, 0::4].size(), device='cuda'))
    pred_boxes[:, 1::4] = torch.max(pred_boxes[:, 1::4], torch.zeros(pred_boxes[:, 1::4].size(), device='cuda'))
    pred_boxes[:, 2::4] = torch.min(pred_boxes[:, 2::4], torch.full(pred_boxes[:, 2::4].size(), boxes.size[0]-1, device='cuda'))
    pred_boxes[:, 3::4] = torch.min(pred_boxes[:, 3::4], torch.full(pred_boxes[:, 3::4].size(), boxes.size[1]-1, device='cuda'))

    return pred_boxes

def shuffle_negatives(negatives, batch_size=None, num_batches=None):
    negatives_to_return = []
    for i in range(len(negatives)):
        if batch_size is None:
            bs = len(negatives[i][0])
        else:
            bs = batch_size
        total_negatives_i = torch.cat(negatives[i])
        if num_batches is None:
            nb = math.ceil(total_negatives_i/bs)
        else:
            nb = num_batches
        shuffled_ids = torch.randperm(len(total_negatives_i))
        negatives_to_return.append([])
        for j in range(nb):
            start_j_index = min(j * bs, len(shuffled_ids))
            end_j_index = min((j + 1) * bs, len(shuffled_ids))
            negatives_to_return[i].append(total_negatives_i[shuffled_ids[start_j_index:end_j_index]])
    return negatives_to_return


def mask_iou(mask_a, mask_b):
    """Calculate the Intersection of Unions (IoUs) between masks.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.
    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`mask_a` and :obj:`mask_b` need to be
    same type.
    The output is same type as the type of the inputs.
    Args:
        mask_a (array): An array whose shape is :math:`(N, H, W)`.
            :math:`N` is the number of masks.
            The dtype should be :obj:`numpy.bool`.
        mask_b (array): An array similar to :obj:`mask_a`,
            whose shape is :math:`(K, H, W)`.
            The dtype should be :obj:`numpy.bool`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th mask in :obj:`mask_a` and :math:`k` th mask \
        in :obj:`mask_b`.
    """

    if mask_a.shape[1:] != mask_b.shape[1:]:
        raise IndexError

    n_mask_a = len(mask_a)
    n_mask_b = len(mask_b)
    iou = np.empty((n_mask_a, n_mask_b), dtype=np.float32)
    for n, m_a in enumerate(mask_a):
        for k, m_b in enumerate(mask_b):
            intersect = np.bitwise_and(m_a, m_b).sum()
            union = np.bitwise_or(m_a, m_b).sum()
            iou[n, k] = intersect / union
    return iou
