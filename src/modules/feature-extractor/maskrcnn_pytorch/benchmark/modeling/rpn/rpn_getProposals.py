# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

#from maskrcnn_benchmark.modeling import registry
from maskrcnn_pytorch.benchmark.modeling import registry
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.rpn.retinanet.retinanet import build_retinanet
from .loss import make_rpn_loss_evaluator
from .anchor_generator import make_anchor_generator
from .inference import make_rpn_postprocessor

from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou, cat_boxlist
from maskrcnn_benchmark.structures.bounding_box import BoxList
import time
import os

import math

class RPNHeadConvRegressor(nn.Module):
    """
    A simple RPN Head for classification and bbox regression
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHeadConvRegressor, self).__init__()
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in [self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        logits = [self.cls_logits(y) for y in x]
        bbox_reg = [self.bbox_pred(y) for y in x]

        return logits, bbox_reg


class RPNHeadFeatureSingleConv(nn.Module):
    """
    Adds a simple RPN Head with one conv to extract the feature
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
        """
        super(RPNHeadFeatureSingleConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

        for l in [self.conv]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

        self.out_channels = in_channels

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        x = [F.relu(self.conv(z)) for z in x]

        return x


@registry.RPN_HEADS.register("SingleConvRPNHead_getProposals")
class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        x_to_ret =[]
        for feature in x:
            feature = F.relu(self.conv(feature))
            x_to_ret.append(feature)
        return x_to_ret

class RPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and outputs 
    RPN proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg, in_channels):
        super(RPNModule, self).__init__()

        self.cfg = cfg.clone()

        anchor_generator = make_anchor_generator(cfg)

        rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]
        head = rpn_head(
            cfg, in_channels, anchor_generator.num_anchors_per_location()[0]
        )

        rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        box_selector_train = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=True)
        box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=False)

        loss_evaluator = make_rpn_loss_evaluator(cfg, rpn_box_coder)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

        self.output_dir = cfg.OUTPUT_DIR
        self.anchors = None

        self.num_classes = self.cfg.MINIBOOTSTRAP.RPN.NUM_CLASSES
        self.iterations = self.cfg.MINIBOOTSTRAP.RPN.ITERATIONS
        self.batch_size = self.cfg.MINIBOOTSTRAP.RPN.BATCH_SIZE
        self.negatives = []
        self.positives = []
        self.current_batch = []
        self.current_batch_size = []
        self.negatives_to_pick = self.cfg.MINIBOOTSTRAP.RPN.NEGATIVES_PER_BATCH

        self.diag_list=[torch.empty(0, dtype=torch.long, device='cuda')]
        for i in range(50):
            if i == 0:
                continue
            else:
                self.diag_list.append(torch.arange(0,i**2, i+1, dtype=torch.long, device='cuda'))

    def forward(self, images, features, targets=None, gt_bbox=None, img_size = None, img_name = None, compute_average_recall_RPN = False, num_classes = None, start_time = None, is_train = None):

        features = self.head(features)
        if self.anchors is None:
            features = features[0][0]
            features_map_size = features.size()
            # Extract feature map info
            self.feat_size = features_map_size[0]
            self.height = features_map_size[1]
            self.width = features_map_size[2]

            # Generate anchors
            self.anchors = self.anchor_generator(images, features)[0][0]
            # COmpute number of anchors for each features tensor
            self.num_classes = int(self.anchors.bbox.size()[0]/(self.height * self.width))

            self.feature_ids = torch.empty((0, 2), dtype=torch.long, device='cuda')
            self.classifiers = torch.empty(0, dtype=torch.uint8, device='cuda')
            # Associate to each feature tensor an id, corresponding to its position and a classifier id corresponding to an anchor value
            for ind in range(0, int(self.anchors.bbox.size()[0])):
                feat_ij = [[int(int(ind/self.num_classes)/self.width), int(int(ind/self.num_classes)%self.width)]]
                self.feature_ids = torch.cat((self.feature_ids, torch.tensor(feat_ij, dtype=torch.long, device='cuda')))
                cls = [ind %self.num_classes]
                self.classifiers = torch.cat((self.classifiers, torch.tensor(cls, dtype=torch.uint8, device='cuda')))
            self.anchors.add_field('feature_id', self.feature_ids)
            self.anchors.add_field('classifier', self.classifiers)
            # Remove features with borders external to the image
            self.visible_anchors = self.anchors.get_field('visibility')
            self.anchors = self.anchors[self.visible_anchors]
            # Avoid computing unuseful regions
            self.still_to_complete = list(range(self.num_classes))
            for i in self.still_to_complete:
                if self.anchors[self.anchors.get_field('classifier') == i].bbox.size()[0] == 0:
                    self.still_to_complete.remove(i)
                    print('Anchor %i does not have visible regions.' %i ,'Removed from the list.') 

            # Initialize batches for minibootstrap
            for i in range(self.num_classes):
                self.negatives.append([])
                self.current_batch.append(0)
                self.current_batch_size.append(0)
                self.positives.append(torch.empty((0, self.feat_size), device='cuda'))
                for j in range(self.iterations):
                    self.negatives[i].append(torch.empty((0, self.feat_size), device='cuda'))

            # Initialize tensors for box regression    
            # Features
            self.X = torch.empty((0, self.feat_size), dtype=torch.float32, device='cuda')
            # Target values
            self.Y = torch.empty((0, 4), dtype=torch.float32, device='cuda')
            # Overlap
            self.O = None
            # Associated classifier (i.e. anchor)
            self.C = torch.empty((0), dtype=torch.float32, device='cuda')
            
        else:
            features = features[0][0]

        anchors_to_return = self.anchors.copy_with_fields(self.anchors.fields())
        # Resize ground truth boxes to anchors dimensions
        gt_bbox = gt_bbox.resize(anchors_to_return.size)
        # Compute anchors-gts ious
        ious = torch.squeeze(boxlist_iou(gt_bbox, anchors_to_return))
        # Associate each anchor with the gt with max iou
        if gt_bbox.bbox.size()[0] > 1:
            ious, ious_index = torch.max(ious, dim=0)
            anchors_to_return.add_field('gt_bbox', gt_bbox.bbox[ious_index])
        else:
            gts = torch.ones((ious.size()[0] ,4), device='cuda') * gt_bbox.bbox[0]
            anchors_to_return.add_field('gt_bbox', gts)
        anchors_to_return.add_field('overlap', ious)

        # Filter all the negatives, i.e. with iou with the gts < 0.3
        negative_anchors_total = anchors_to_return[ious < 0.3]

        indices_to_remove = []
        for i in self.still_to_complete:
            # Filter negatives for the i-th anchor
            anchors_i = negative_anchors_total[negative_anchors_total.get_field('classifier')==i]
            # Sample negatives, according to minibootstrap parameters
            if anchors_i.bbox.size()[0] > self.negatives_to_pick:
                anchors_i = anchors_i[torch.randint(anchors_i.bbox.size()[0], (self.negatives_to_pick,))]
            # Compute their id, i.e. position in the features map
            ids = anchors_i.get_field('feature_id')
            ids_size = ids.size()[0]
            # Compute at most how many negatives to add to each batch
            reg_to_add = math.ceil(self.negatives_to_pick/self.iterations)
            # Initialize index of chosen negatives among all the negatives to pick
            ind_to_add = 0
            for b in range(self.current_batch[i], self.iterations):
                # If the batch is full, start from the subsequent
                if self.negatives[i][b].size()[0] >= self.batch_size:
                    self.current_batch[i] += 1
                    if self.current_batch[i] >= self.iterations:
                        indices_to_remove.append(i)
                    continue
                else:
                    # Compute the end index of negatives to add to the batch
                    end_interval = int(ind_to_add + min(reg_to_add, self.batch_size - self.negatives[i][b].size()[0], self.negatives_to_pick - ind_to_add, ids_size -ind_to_add))
                    # Extract features corresponding to the ids and add them to the ids
                    # Diagonal choice done for computational efficiency
                    feat = torch.index_select(features, 1, ids[ind_to_add:end_interval, 0])
                    feat = torch.index_select(feat, 2, ids[ind_to_add:end_interval, 1]).permute(1,2,0).view((end_interval-ind_to_add)**2,self.feat_size)
                    try:
                        feat = feat[self.diag_list[end_interval-ind_to_add]]
                    except:
                        feat = feat[list(range(0,(end_interval-ind_to_add)**2+(end_interval-ind_to_add)-1, (end_interval-ind_to_add)+1))]
                    self.negatives[i][b] = torch.cat((self.negatives[i][b], feat))
                    # Update indices
                    ind_to_add = end_interval
                    if ind_to_add == self.negatives_to_pick:
                        break
        # Check to avoid unuseful computations
        for index in indices_to_remove:
            self.still_to_complete.remove(index)

        # Select all the positives with iou with the gts > 0.7
        positive_anchors = anchors_to_return[anchors_to_return.get_field('overlap') > 0.7]

        # Add to the positives anchors with max iou with a gt, if the gt doesn't have associated anchors with iou > 0.7
        for elem in gt_bbox.bbox:
            if elem in positive_anchors.get_field('gt_bbox'):
                continue
            else:
                elem = elem.unsqueeze(0)
                # Find indices where there are anchors associated to this gt_bbox
                indices, _= torch.min(torch.eq(anchors_to_return.get_field('gt_bbox'), elem.repeat(anchors_to_return.bbox.size()[0],1)), dim=1, keepdim=True)
                # Find max overlap with this gt_bbox
                values, _ = torch.max(anchors_to_return[indices.squeeze()].get_field('overlap'), 0)
                positives_i = anchors_to_return[indices.squeeze()]
                positives_i = positives_i[positives_i.get_field('overlap') == values.item()]
                positive_anchors = cat_boxlist([positive_anchors, positives_i])
                
        # Find anchors associated to the positives, to avoid unuseful computation
        pos_inds = torch.unique(positive_anchors.get_field('classifier'))
        for i in pos_inds:
            anchors_i = positive_anchors[positive_anchors.get_field('classifier')==i]
            ids = anchors_i.get_field('feature_id')
            ids_size = ids.size()[0]
            feat = torch.index_select(features, 1, ids[:,0])
            feat = torch.index_select(feat, 2, ids[:,1]).permute(1,2,0).view(ids_size**2,self.feat_size)
            try:
                feat = feat[self.diag_list[ids_size]]
            except:
                feat = feat[list(range(0,ids_size**2+ids_size-1, ids_size+1))]
            # Add positive features for the i-th anchor to the i-th positives list
            self.positives[i] = torch.cat((self.positives[i], feat))

            # COXY computation for regressors
            ex_boxes = anchors_i.bbox
            gt_boxes = anchors_i.get_field('gt_bbox')
            self.X = torch.cat((self.X, feat))
            self.C = torch.cat((self.C, torch.full((ids_size,1), i, dtype=torch.float32, device='cuda')))

            src_w = ex_boxes[:,2] - ex_boxes[:,0] + 1
            src_h = ex_boxes[:,3] - ex_boxes[:,1] + 1
            src_ctr_x = ex_boxes[:,0] + 0.5 * src_w
            src_ctr_y = ex_boxes[:,1] + 0.5 * src_h

            gt_w = gt_boxes[:,2] - gt_boxes[:,0] + 1
            gt_h = gt_boxes[:,3] - gt_boxes[:,1] + 1
            gt_ctr_x = gt_boxes[:,0] + 0.5 * gt_w
            gt_ctr_y = gt_boxes[:,1] + 0.5 * gt_h

            dst_ctr_x = (gt_ctr_x - src_ctr_x) / src_w
            dst_ctr_y = (gt_ctr_y - src_ctr_y) / src_h
            dst_scl_w = torch.log(gt_w / src_w)
            dst_scl_h = torch.log(gt_h / src_h)

            target = torch.stack((dst_ctr_x, dst_ctr_y, dst_scl_w, dst_scl_h), dim=1)
            self.Y = torch.cat((self.Y, target), dim=0)

        return {}, {}, 0

def build_rpn(cfg, in_channels):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    if cfg.MODEL.RETINANET_ON:
        return build_retinanet(cfg, in_channels)

    return RPNModule(cfg, in_channels)
