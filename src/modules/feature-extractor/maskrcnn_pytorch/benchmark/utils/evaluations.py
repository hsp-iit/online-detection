import numpy as np
import torch

def getIntersectionBox(box1, box2):
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]

    xmin = max(x1min, x2min)
    ymin = max(y1min, y2min)
    xmax = min(x1max, x2max)
    ymax = min(y1max, y2max)

    intersectionBox = xmin, ymin, xmax, ymax
    return intersectionBox

def compute_overlap(gt, pred):
    xmin, ymin, xmax, ymax = getIntersectionBox(gt, pred)
    intersection_area = (xmax - xmin + 1) * (ymax - ymin + 1)
    if (xmax - xmin + 1) <= 0 or (ymax - ymin + 1) <= 0:
        # No overlap
        return 0
    gt_area = (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)
    pred_area = (pred[2] - pred[0] + 1) * (pred[3] - pred[1] + 1)
    overlap = intersection_area / (gt_area + pred_area - intersection_area)
    return overlap


def compute_overlap_torch(gt, prop):

    gt_tens = torch.ones((prop.size()[0], 4), device='cuda') * gt
    xmin = torch.max(gt_tens[:, 0], prop[:,0])
    ymin = torch.max(gt_tens[:, 1], prop[:,1])
    xmax = torch.min(gt_tens[:, 2], prop[:,2])
    ymax = torch.min(gt_tens[:, 3], prop[:,3])
    intersection_area = (xmax - xmin + 1) * (ymax - ymin + 1)
    #print(xmin)
    gt_area = (gt_tens[:, 2] - gt_tens[:, 0] + 1) * (gt_tens[:, 3] - gt_tens[:, 1] + 1)
    pred_area = (prop[:, 2] - prop[:, 0] + 1) * (prop[:, 3] - prop[:, 1] + 1)
    overlap = intersection_area / (gt_area + pred_area - intersection_area)
    overlap = torch.where((xmax - xmin + 1) > 0, overlap, torch.zeros(overlap.size(), device='cuda'))
    overlap = torch.where((ymax - ymin + 1) > 0, overlap, torch.zeros(overlap.size(), device='cuda'))

    return overlap
