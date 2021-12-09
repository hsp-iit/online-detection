# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator

import os
import numpy as np

from mrcnn_modified.utils.evaluations import compute_overlap_torch
import math

class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
        self.cfg = cfg
        self.training_device = self.cfg.MINIBOOTSTRAP.DETECTOR.FEATURES_DEVICE

        try:
            self.save_features = self.cfg.SAVE_FEATURES_DETECTOR
        except:
            self.save_features = False

        self.initialize_online_detection_params()

    def initialize_online_detection_params(self, num_classes=0, num_images=None):
        self.num_classes = num_classes if num_classes else self.cfg.MINIBOOTSTRAP.DETECTOR.NUM_CLASSES
        self.iterations = self.cfg.MINIBOOTSTRAP.DETECTOR.ITERATIONS
        self.batch_size = self.cfg.MINIBOOTSTRAP.DETECTOR.BATCH_SIZE
        self.compute_gt_positives = self.cfg.MINIBOOTSTRAP.DETECTOR.EXTRACT_ONLY_GT_POSITIVES
        self.shuffle_negatives = self.cfg.MINIBOOTSTRAP.DETECTOR.SHUFFLE_NEGATIVES
        self.incremental_train = self.cfg.DEMO.INCREMENTAL_TRAIN
        if self.compute_gt_positives:
            self.positives = []
        self.negatives = []
        self.current_batch = []
        self.current_batch_size = []
        if self.incremental_train:
            self.buffer_negatives =[]
        for i in range(self.num_classes):
            if self.compute_gt_positives:
                self.positives.append([torch.empty((0, self.feature_extractor.out_channels), device=self.training_device)])
            if not self.shuffle_negatives:
                self.negatives.append([])
                self.current_batch.append(0)
                self.current_batch_size.append(0)
                for j in range(self.iterations):
                    self.negatives[i].append(torch.empty((0, self.feature_extractor.out_channels), device=self.training_device))
            else:
                if self.incremental_train:
                    self.negatives.append([])
                    self.buffer_negatives.append([])
                else:
                    self.negatives.append([torch.empty((0, self.feature_extractor.out_channels), device=self.training_device)])

        self.negatives_to_pick = None

        if num_images is not None:
            self.cfg.NUM_IMAGES = num_images
        
        self.neg_iou_thresh = self.cfg.MINIBOOTSTRAP.DETECTOR.NEG_IOU_THRESH
        self.still_to_complete = list(range(self.num_classes))

        self.reg_min_overlap = self.cfg.REGRESSORS.MIN_OVERLAP

        # Regressor features
        self.X = [torch.empty((0, self.feature_extractor.out_channels), dtype=torch.float32, device=self.training_device)]
        # Regressor target values
        self.Y = [torch.empty((0, 4), dtype=torch.float32, device=self.training_device)]
        # Regressor overlap amounts
        self.O = None
        # Regressor classes
        self.C = [torch.empty((0), dtype=torch.float32, device=self.training_device)]

        self.test_boxes = []

    def add_new_class(self):
        self.still_to_complete.append(self.num_classes)
        self.num_classes += 1
        self.negatives.append([])
        self.current_batch.append(0)
        self.current_batch_size.append(0)
        if self.compute_gt_positives:
            self.positives.append([torch.empty((0, self.feature_extractor.out_channels), device=self.training_device)])
        for j in range(self.iterations):
            self.negatives[len(self.negatives)-1].append(torch.empty((0, self.feature_extractor.out_channels), device=self.training_device))

    def forward(self, features, proposals, gt_bbox = None, gt_label = None, img_size= None, gt_labels_list=None, is_train = True, result_dir = None):
        if is_train:
            return self.forward_train(features, proposals, gt_bbox=gt_bbox, gt_label=gt_label, img_size=img_size, gt_labels_list=gt_labels_list, result_dir=result_dir)
        else:
            return self.forward_test(features, proposals, gt_bbox=gt_bbox, gt_label=gt_label, img_size=img_size, gt_labels_list=gt_labels_list)

    def forward_train(self, features, proposals, gt_bbox=None, gt_label=None, img_size=None, gt_labels_list=None, result_dir=None):

        if self.negatives_to_pick is None:
            self.negatives_to_pick = math.ceil((self.batch_size*self.iterations)/self.cfg.NUM_IMAGES)

        # Extract features that will be fed to the final classifier.
        feat = self.feature_extractor(features, proposals)
        x = self.avgpool(feat)
        x = x.view(x.size(0), -1)
        proposals[0] = proposals[0].resize((img_size[0], img_size[1]))
        gt_bbox = gt_bbox.resize((img_size[0], img_size[1]))

        # Compute proposed bboxes and gt bboxes
        arr_proposals = proposals[0].bbox
        arr_gt_bbox = gt_bbox.bbox

        arr_proposals[:, 2] = torch.clamp(arr_proposals[:, 2], 0, img_size[0]-1)
        arr_proposals[:, 0] = torch.clamp(arr_proposals[:, 0], 0, img_size[0]-1)
        arr_proposals[:, 3] = torch.clamp(arr_proposals[:, 3], 0, img_size[1]-1)
        arr_proposals[:, 1] = torch.clamp(arr_proposals[:, 1], 0, img_size[1]-1)

        arr_gt_bbox[:, 2] = torch.clamp(arr_gt_bbox[:, 2], 0, img_size[0]-1)
        arr_gt_bbox[:, 0] = torch.clamp(arr_gt_bbox[:, 0], 0, img_size[0]-1)
        arr_gt_bbox[:, 3] = torch.clamp(arr_gt_bbox[:, 3], 0, img_size[1]-1)
        arr_gt_bbox[:, 1] = torch.clamp(arr_gt_bbox[:, 1], 0, img_size[1]-1)

        # Count gt bboxes
        if gt_label is not None:
            len_gt = gt_label.size()[0]
        else:
            len_gt = 0
        # Count num of proposals
        num_proposals = arr_proposals.size()[0]- len_gt
        # Specify the classes of the ground truth boxes, then set the classes of the not-gt proposals to 0
        if gt_label is not None:
            arr_class = torch.cat((gt_label, torch.zeros((num_proposals, 1), device='cuda', dtype=torch.uint8)), dim=0)
        else:
            arr_class = torch.zeros((num_proposals,1), device='cuda')
        # Initialize overlaps with gts to 0
        overlap = torch.zeros((arr_proposals.size()[0], self.num_classes), dtype=torch.float, device='cuda')
        # Initialize tensors to keep track of the gt with max overlap for each proposal
        associated_gt_id = torch.full((arr_proposals.size()[0],), -1, dtype=torch.int, device='cuda')
        max_iou_gt = torch.zeros((arr_proposals.size()[0],), dtype=torch.float, device='cuda')
        # Compute max overlap of each proposal with each class
        for j in range(arr_gt_bbox.size()[0]):
            overlap_j = compute_overlap_torch(arr_gt_bbox[j], arr_proposals)
            overlap[:, gt_labels_list[j]-1] = torch.max(overlap[:, gt_labels_list[j]-1], overlap_j)
            ids_to_update = torch.where(overlap_j > max_iou_gt)
            associated_gt_id[ids_to_update] = j
            max_iou_gt[ids_to_update] = overlap_j[ids_to_update]
            

        # Loop on all the gt boxes
        for i in range(len(gt_labels_list)):
            if self.compute_gt_positives:
                # Concatenate each gt to the positive tensor for its corresponding class
                if self.training_device is 'cpu':
                    self.positives[gt_labels_list[i]-1][len(self.positives[gt_labels_list[i]-1]) - 1] = torch.cat((self.positives[gt_labels_list[i]-1][len(self.positives[gt_labels_list[i]-1]) - 1], x[i].view(-1, self.feature_extractor.out_channels).cpu()))
                else:
                    self.positives[gt_labels_list[i]-1][len(self.positives[gt_labels_list[i]-1]) - 1] = torch.cat((self.positives[gt_labels_list[i]-1][len(self.positives[gt_labels_list[i]-1]) - 1], x[i].view(-1, self.feature_extractor.out_channels)))
                if self.positives[gt_labels_list[i]-1][len(self.positives[gt_labels_list[i]-1]) - 1].size()[0] >= self.batch_size:
                    if self.save_features:
                        path_to_save = os.path.join(result_dir, 'features_detector', 'positives_cl_{}_batch_{}'.format(gt_labels_list[i]-1, len(self.positives[gt_labels_list[i]-1]) - 1))
                        torch.save(self.positives[gt_labels_list[i]-1][len(self.positives[gt_labels_list[i]-1]) - 1], path_to_save)
                        self.positives[gt_labels_list[i]-1][len(self.positives[gt_labels_list[i]-1]) - 1] = torch.empty((0, self.feature_extractor.out_channels), device=self.training_device)
                    self.positives[gt_labels_list[i]-1].append(torch.empty((0, self.feature_extractor.out_channels), device=self.training_device))

            # Extract regressor positives, i.e. with overlap > self.reg_min_overlap and with proposed boxes associated to that gt
            pos_ids = overlap[:,gt_labels_list[i]-1] > self.reg_min_overlap
            pos_ids = pos_ids & torch.eq(associated_gt_id, i)
            regr_positives_i = x[pos_ids].view(-1, self.feature_extractor.out_channels)

            # Compute targets
            ex_boxes = arr_proposals[pos_ids].view(-1, 4)
            gt_boxes = torch.ones(ex_boxes.size(), device='cuda') * arr_proposals[i].view(-1, 4)

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

            if self.training_device is 'cpu':
                self.Y[len(self.Y)-1] = torch.cat((self.Y[len(self.Y)-1], target.cpu()), dim=0)
                # Add class and features to C and X
                self.C[len(self.C)-1] = torch.cat((self.C[len(self.C)-1], torch.full((torch.sum(pos_ids), 1), gt_labels_list[i])))
                self.X[len(self.X)-1] = torch.cat((self.X[len(self.X)-1], regr_positives_i.cpu()))
            else:
                self.Y[len(self.Y)-1] = torch.cat((self.Y[len(self.Y)-1], target), dim=0)
                # Add class and features to C and X
                self.C[len(self.C)-1] = torch.cat((self.C[len(self.C)-1], torch.full((torch.sum(pos_ids), 1), gt_labels_list[i], device='cuda')))
                self.X[len(self.X)-1] = torch.cat((self.X[len(self.X)-1], regr_positives_i))
            if self.X[len(self.X)-1].size()[0] >= self.batch_size:
                if self.save_features:
                    path_to_save = os.path.join(result_dir, 'features_detector','reg_x_batch_{}'.format(len(self.X)-1))
                    torch.save(self.X[len(self.X)-1], path_to_save)
                    self.X[len(self.X)-1] = torch.empty((0, self.feature_extractor.out_channels), dtype=torch.float32, device=self.training_device)

                    path_to_save = os.path.join(result_dir, 'features_detector','reg_c_batch_{}'.format(len(self.C)-1))
                    torch.save(self.C[len(self.C)-1], path_to_save)
                    self.C[len(self.C)-1] = torch.empty((0), dtype=torch.float32, device=self.training_device)

                    path_to_save = os.path.join(result_dir, 'features_detector','reg_y_batch_{}'.format(len(self.Y)-1))
                    torch.save(self.Y[len(self.Y)-1], path_to_save)
                    self.Y[len(self.Y)-1] = torch.empty((0, 4), dtype=torch.float32, device=self.training_device)

                self.X.append(torch.empty((0, self.feature_extractor.out_channels), dtype=torch.float32, device=self.training_device))
                self.C.append(torch.empty((0), dtype=torch.float32, device=self.training_device))
                self.Y.append(torch.empty((0, 4), dtype=torch.float32, device=self.training_device))

        if not self.shuffle_negatives:
            # Fill batches for minibootstrap
            indices_to_remove = []
            # Loop on all the classes that doesn't have full batches
            for i in self.still_to_complete:
                # Add random negatives, if there isn't a gt corresponding to that class
                if i+1 not in gt_labels_list:
                    neg_i = x[torch.randint(x.size()[0], (self.negatives_to_pick,))].view(-1, self.feature_extractor.out_channels)
                # Add random examples with iou < 0.3 otherwise
                else:
                    neg_i = x[overlap[:,i] < self.neg_iou_thresh].view(-1, self.feature_extractor.out_channels)
                    if neg_i.size()[0] > 0:
                        neg_i = neg_i[torch.randint(neg_i.size()[0], (self.negatives_to_pick,))].view(-1, self.feature_extractor.out_channels)
                # Add negatives splitting them into batches
                neg_to_add = math.ceil(self.negatives_to_pick/self.iterations)
                ind_to_add = 0
                for b in range(self.current_batch[i], self.iterations):
                    if self.negatives[i][b].size()[0] >= self.batch_size:
                        # If features must be saved, save full batches and replace the batch in gpu with an empty tensor
                        if self.save_features:
                            path_to_save = os.path.join(result_dir, 'features_detector', 'negatives_cl_{}_batch_{}'.format(i, b))
                            torch.save(self.negatives[i][b], path_to_save)
                            self.negatives[i][b] = torch.empty((0, self.feature_extractor.out_channels), device=self.training_device)
                        self.current_batch[i] += 1
                        if self.current_batch[i] >= self.iterations:
                            indices_to_remove.append(i)
                        continue
                    else:
                        end_interval = int(ind_to_add + min(neg_to_add, self.batch_size - self.negatives[i][b].size()[0], self.negatives_to_pick - ind_to_add))
                        if self.training_device is 'cpu':
                            self.negatives[i][b] = torch.cat((self.negatives[i][b], neg_i[ind_to_add:end_interval].view(-1, self.feature_extractor.out_channels).cpu()))
                        else:
                            self.negatives[i][b] = torch.cat((self.negatives[i][b], neg_i[ind_to_add:end_interval].view(-1, self.feature_extractor.out_channels)))
                        ind_to_add = end_interval
                        if ind_to_add == self.negatives_to_pick:
                            break
            for index in indices_to_remove:
                self.still_to_complete.remove(index)
        else:
            for i in range(len(self.negatives)):
                # Add random negatives, if there isn't a gt corresponding to that class
                if (i + 1 not in gt_labels_list) or self.incremental_train:
                    neg_i = x[torch.randint(x.size()[0], (self.negatives_to_pick,))].view(-1, self.feature_extractor.out_channels)
                    if self.incremental_train:
                        self.buffer_negatives[i].append(neg_i)
                # Add random examples with iou < 0.3 otherwise
                if (i + 1 in gt_labels_list) or self.incremental_train:
                    neg_i = x[overlap[:, i] < self.neg_iou_thresh].view(-1, self.feature_extractor.out_channels)
                    if neg_i.size()[0] > 0:
                        neg_i = neg_i[torch.randint(neg_i.size()[0], (self.negatives_to_pick,))].view(-1, self.feature_extractor.out_channels)
                if self.incremental_train:
                    self.negatives[i].append(neg_i)
                else:
                    if self.training_device is 'cpu':
                        self.negatives[i][len(self.negatives[i]) - 1] = torch.cat((self.negatives[i][len(self.negatives[i]) - 1], neg_i.cpu()))
                    else:
                        self.negatives[i][len(self.negatives[i]) - 1] = torch.cat((self.negatives[i][len(self.negatives[i]) - 1], neg_i))
                if self.negatives[i][len(self.negatives[i]) - 1].size()[0] >= self.batch_size:
                    if self.save_features:
                        path_to_save = os.path.join(result_dir, 'features_detector', 'negatives_cl_{}_batch_{}'.format(i, len(self.negatives[i]) - 1))
                        torch.save(self.negatives[i][len(self.negatives[i]) - 1], path_to_save)
                        self.negatives[i][len(self.negatives[i]) - 1] = torch.empty((0, self.feature_extractor.out_channels), device=self.training_device)
                    self.negatives[i].append(torch.empty((0, self.feature_extractor.out_channels), device=self.training_device))

        return feat, None, None


    def forward_test(self, features, proposals, gt_bbox = None, gt_label = None, img_size= None, gt_labels_list=None):

        # Extract features that will be fed to the final classifier.
        x = self.feature_extractor(features, proposals)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        proposals[0] = proposals[0].resize((img_size[0], img_size[1]))
        gt_bbox = gt_bbox.resize((img_size[0], img_size[1]))
        
        # Compute proposed bboxes and gt bboxes
        arr_proposals = proposals[0].bbox
        arr_gt_bbox = gt_bbox.bbox

        arr_proposals[:, 2] = torch.clamp(arr_proposals[:, 2], 0, img_size[0]-1)
        arr_proposals[:, 0] = torch.clamp(arr_proposals[:, 0], 0, img_size[0]-1)
        arr_proposals[:, 3] = torch.clamp(arr_proposals[:, 3], 0, img_size[1]-1)
        arr_proposals[:, 1] = torch.clamp(arr_proposals[:, 1], 0, img_size[1]-1)

        arr_gt_bbox[:, 2] = torch.clamp(arr_gt_bbox[:, 2], 0, img_size[0]-1)
        arr_gt_bbox[:, 0] = torch.clamp(arr_gt_bbox[:, 0], 0, img_size[0]-1)
        arr_gt_bbox[:, 3] = torch.clamp(arr_gt_bbox[:, 3], 0, img_size[1]-1)
        arr_gt_bbox[:, 1] = torch.clamp(arr_gt_bbox[:, 1], 0, img_size[1]-1)

        # Count gt bboxes
        if gt_label is not None:
            len_gt = gt_label.size()[0]
        else:
            len_gt = 0
        # Count num of proposals
        num_proposals = arr_proposals.size()[0]- len_gt
        # Specify the classes of the ground truth boxes, then set the classes of the not-gt proposals to 0
        if gt_label is not None:
            arr_class = torch.cat((gt_label, torch.zeros((num_proposals, 1), device='cuda', dtype=torch.uint8)), dim=0)
        else:
            arr_class = torch.zeros((num_proposals,1), device='cuda')
        # Signal if the box is a gt or not
        arr_gt = arr_class > 0
        self.test_boxes.append({'boxes': arr_proposals.cpu().numpy(), 'feat': x.cpu().numpy(), 'gt': arr_gt.cpu().numpy(), 'img_size': np.array(img_size)})

        return None, None, None


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
