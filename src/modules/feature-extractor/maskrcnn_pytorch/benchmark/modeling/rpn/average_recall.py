import torch
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

def compute_average_recall(ground_truths, proposals):
    match_quality_matrix = boxlist_iou(ground_truths, proposals)
    highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
    highest_quality_foreach_gt -= 0.5
    AR = 2*torch.mean(torch.max(highest_quality_foreach_gt, torch.zeros(highest_quality_foreach_gt.size(), device='cuda'))).item()
    return AR
    
