# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.box_head_getProposals import build_roi_box_head
from .mask_head.mask_head_getProposals import build_roi_mask_head

from maskrcnn_benchmark.modeling.roi_heads.keypoint_head.keypoint_head import build_roi_keypoint_head
from maskrcnn_benchmark.structures.bounding_box import BoxList

class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, gt_bbox=None, gt_label=None, img_size=[0,0], gt_labels_list=None, is_train=True, result_dir=None, extract_features_segmentation=False):
        losses = {}
        x, detections, loss_box = self.box(features, proposals, gt_bbox=gt_bbox, gt_label=gt_label, img_size=img_size, gt_labels_list=gt_labels_list, is_train=is_train, result_dir=result_dir)

        if self.cfg.MODEL.MASK_ON and extract_features_segmentation:
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                mask_features = x
            else:
                mask_features = features
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_mask = self.mask(mask_features, proposals, gt_labels_list, gt_bbox, result_dir=result_dir)

        return x, detections, losses


def build_roi_heads(cfg, in_channels):
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
