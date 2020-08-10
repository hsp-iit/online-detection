# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from maskrcnn_benchmark.modeling.backbone import build_backbone
from maskrcnn_pytorch.benchmark.modeling.rpn.rpn import build_rpn
#from maskrcnn_benchmark.modeling.rpn.rpn import build_rpn
from ..roi_heads.roi_heads_getProposals import build_roi_heads

import time

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, image_name, targets=None, gt_bbox = None, gt_label = None, img_size = [0,0], start_time = None, compute_average_recall_RPN=False, num_classes=30, gt_labels_list = None, is_train = True):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        #print('Generalized',gt_bbox.bbox)
        # TODO ms-thesis-segmentation Adapt here to only use the mask branch and the passed values
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        # TODO ms-thesis-segmentation adapt here
        #features = {}
        #proposals = {}
        #  TODO ms-thesis-segmentation add the ground truth proposal to the vector
        #quit()
        proposals, proposal_losses, average_recall_RPN = self.rpn(images, features, gt_bbox.resize((images.image_sizes[0][1], images.image_sizes[0][0])), compute_average_recall_RPN=compute_average_recall_RPN)
        #print(average_recall_RPN)
        #return average_recall_RPN       #TODO remove this
        #print(proposals, gt_bbox.resize((images.image_sizes[0][1], images.image_sizes[0][0])))
        if gt_bbox is not None:
            # Resize the ground truth boxes to the correct format
            width, height = proposals[0].size
            gt_bbox = gt_bbox.resize((width, height))
            # Add the ground truth proposals to the proposal vector
            proposals[0].bbox = torch.cat((gt_bbox.bbox, proposals[0].bbox), 0)
            #proposals[0].extra_fields['objectness'] = torch.cat((1.0 * torch.ones(gt_bbox.bbox.size()[0]), proposals[0].extra_fields['objectness']), 0)

            proposals[0].extra_fields['objectness'] = torch.cat((1.0 * torch.ones(gt_bbox.bbox.size()[0], device="cuda"), proposals[0].extra_fields['objectness']), 0)

        #print('Generalized after resize',gt_bbox.bbox)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, image_name, targets, gt_bbox = gt_bbox, gt_label= gt_label, img_size=img_size, start_time=start_time, num_classes=num_classes, gt_labels_list = gt_labels_list, is_train = is_train)
            #print("Here")
        #else:
            # RPN-only models don't have roi_heads
            #x = features
            #result = proposals
            #detector_losses = {}

        #if self.training:
            #losses = {}
            #losses.update(detector_losses)
            #losses.update(proposal_losses)
            #return losses

        return average_recall_RPN

