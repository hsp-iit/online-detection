# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F


from maskrcnn_benchmark.structures.bounding_box import BoxList

from .roi_mask_feature_extractors import make_roi_mask_feature_extractor
from .roi_mask_predictors import make_roi_mask_predictor
from .inference import make_roi_mask_post_processor
from .loss import make_roi_mask_loss_evaluator

import os

def project_masks_on_boxes(segmentation_masks, proposals, discretization_size):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    masks = []
    M = discretization_size
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    assert segmentation_masks.size == proposals.size, "{}, {}".format(
        segmentation_masks, proposals
    )

    # FIXME: CPU computation bottleneck, this should be parallelized
    proposals = proposals.bbox.to(torch.device("cpu"))
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation.
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M, M))
        mask = scaled_mask.get_mask_tensor()
        masks.append(mask)
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0).to(device, dtype=torch.float32)


class ROIMaskHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIMaskHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_mask_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_mask_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_mask_post_processor(cfg)
        self.loss_evaluator = make_roi_mask_loss_evaluator(cfg)
        try:
            self.save_features = self.cfg.SAVE_FEATURES_DETECTOR
        except:
            self.save_features = False

        self.training_device = self.cfg.SEGMENTATION.FEATURES_DEVICE

        self.initialize_online_segmentation_params()

    def initialize_online_segmentation_params(self, num_classes=0):
        self.num_classes = num_classes if num_classes else self.cfg.MINIBOOTSTRAP.DETECTOR.NUM_CLASSES
        self.batch_size = self.cfg.SEGMENTATION.BATCH_SIZE
        self.positives = []
        self.negatives = []
        for i in range(self.num_classes):
            self.positives.append([torch.empty((0, self.predictor.mask_fcn_logits.in_channels), device='cuda')])
            self.negatives.append([torch.empty((0, self.predictor.mask_fcn_logits.in_channels), device='cuda')])

        self.sampling_factor = self.cfg.SEGMENTATION.SAMPLING_FACTOR

    def add_new_class(self):
        self.num_classes += 1
        self.positives.append([torch.empty((0, self.predictor.mask_fcn_logits.in_channels), device='cuda')])
        self.negatives.append([torch.empty((0, self.predictor.mask_fcn_logits.in_channels), device='cuda')])

    def forward(self, features, proposals, gt_labels_list, gt_bbox, targets=None, result_dir=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features[:len(gt_labels_list)]
        else:
            x = self.feature_extractor(features, proposals)[:len(gt_labels_list)]

        if self.cfg.MODEL.ROI_MASK_HEAD['FEATURE_EXTRACTOR'] == 'ResNet50Conv5ROIFeatureExtractor':
            masks_features = F.relu(self.predictor.conv5_mask(x))

        masks_gts = project_masks_on_boxes(gt_bbox.get_field('masks'), gt_bbox, masks_features.size()[2])
        for i in range(len(masks_features)):
            mask_features = masks_features[i].permute(1, 2, 0).view(-1, masks_features.size()[1])
            masks_gt = masks_gts[i].view(mask_features.size()[0])
            # Select pixels positives indices, i.e. where the gt mask value is >= 0.5
            positives_indices = torch.where(masks_gt >= 0.5)[0]
            # Sample the required fraction of indices
            if self.sampling_factor < 1.0:
                sampled_indices = torch.randperm(len(positives_indices))[:int(self.sampling_factor*len(positives_indices))]
                positives_indices = positives_indices[sampled_indices]
            # Add positives of the given mask to the positives list of the corresponding object class
            self.positives[gt_labels_list[i]-1][len(self.positives[gt_labels_list[i]-1]) - 1] = torch.cat((self.positives[gt_labels_list[i]-1][len(self.positives[gt_labels_list[i]-1]) - 1], mask_features[positives_indices]))
            # Manage full batches of features
            if self.positives[gt_labels_list[i]-1][len(self.positives[gt_labels_list[i]-1]) - 1].size()[0] >= self.batch_size:
                if self.save_features:
                    path_to_save = os.path.join(result_dir, 'features_segmentation', 'positives_cl_{}_batch_{}'.format(gt_labels_list[i]-1, len(self.positives[gt_labels_list[i]-1]) - 1))
                    torch.save(self.positives[gt_labels_list[i]-1][len(self.positives[gt_labels_list[i]-1]) - 1], path_to_save)
                    self.positives[gt_labels_list[i]-1][len(self.positives[gt_labels_list[i]-1]) - 1] = torch.empty((0, self.predictor.conv5_mask.out_channels), device=self.training_device)
                if self.training_device == 'cpu':
                    self.positives[gt_labels_list[i] - 1][len(self.positives[gt_labels_list[i] - 1]) - 1] = self.positives[gt_labels_list[i] - 1][len(self.positives[gt_labels_list[i] - 1]) - 1].to('cpu')
                self.positives[gt_labels_list[i]-1].append(torch.empty((0, self.predictor.conv5_mask.out_channels), device='cuda'))
            # Repeat the procedure done for positive features with the negatives
            negatives_indices = torch.where(masks_gt < 0.5)[0]
            if self.sampling_factor < 1.0:
                sampled_indices = torch.randperm(len(negatives_indices))[:int(self.sampling_factor*len(negatives_indices))]
                negatives_indices = negatives_indices[sampled_indices]
            self.negatives[gt_labels_list[i]-1][len(self.negatives[gt_labels_list[i]-1]) - 1] = torch.cat((self.negatives[gt_labels_list[i]-1][len(self.negatives[gt_labels_list[i]-1]) - 1], mask_features[negatives_indices]))
            if self.negatives[gt_labels_list[i]-1][len(self.negatives[gt_labels_list[i]-1]) - 1].size()[0] >= self.batch_size:
                if self.save_features:
                    path_to_save = os.path.join(result_dir, 'features_segmentation', 'negatives_cl_{}_batch_{}'.format(gt_labels_list[i]-1, len(self.negatives[gt_labels_list[i]-1]) - 1))
                    torch.save(self.negatives[gt_labels_list[i]-1][len(self.negatives[gt_labels_list[i]-1]) - 1], path_to_save)
                    self.negatives[gt_labels_list[i]-1][len(self.negatives[gt_labels_list[i]-1]) - 1] = torch.empty((0, self.predictor.conv5_mask.out_channels), device=self.training_device)
                if self.training_device == 'cpu':
                    self.negatives[gt_labels_list[i]-1][len(self.negatives[gt_labels_list[i]-1])-1] = self.negatives[gt_labels_list[i]-1][len(self.negatives[gt_labels_list[i]-1])-1].to('cpu')
                self.negatives[gt_labels_list[i]-1].append(torch.empty((0, self.predictor.conv5_mask.out_channels), device='cuda'))

        return None, None, None


def build_roi_mask_head(cfg, in_channels):
    return ROIMaskHead(cfg, in_channels)
