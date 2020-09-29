import numpy as np
import torch

def compute_overlap_torch(gt, prop):

    gt_tens = torch.ones((prop.size()[0], 4), device='cuda') * gt
    xmin = torch.max(gt_tens[:, 0], prop[:,0])
    ymin = torch.max(gt_tens[:, 1], prop[:,1])
    xmax = torch.min(gt_tens[:, 2], prop[:,2])
    ymax = torch.min(gt_tens[:, 3], prop[:,3])
    intersection_area = (xmax - xmin + 1) * (ymax - ymin + 1)
    gt_area = (gt_tens[:, 2] - gt_tens[:, 0] + 1) * (gt_tens[:, 3] - gt_tens[:, 1] + 1)
    pred_area = (prop[:, 2] - prop[:, 0] + 1) * (prop[:, 3] - prop[:, 1] + 1)
    overlap = intersection_area / (gt_area + pred_area - intersection_area)
    overlap = torch.where((xmax - xmin + 1) > 0, overlap, torch.zeros(overlap.size(), device='cuda'))
    overlap = torch.where((ymax - ymin + 1) > 0, overlap, torch.zeros(overlap.size(), device='cuda'))

    return overlap
