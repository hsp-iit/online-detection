# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

#from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head
from maskrcnn_benchmark.modeling.roi_heads.keypoint_head.keypoint_head import build_roi_keypoint_head
from maskrcnn_benchmark.structures.bounding_box import BoxList

import time

# to save objects
import pickle
# to save as .mat
# TODO ms-thesis-segmentation important!! install scipy to your virtual environment
import scipy.io
import os


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
        #if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            #self.keypoint.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, image_name_path, targets=None, img_size = [0,0], transformed_size = None, detections = None, use_gt = False):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        # TODO ms-thesis-segmentation adapt here to only use the mask branch
        #x, detections, loss_box = self.box(features, proposals, targets)
        #losses.update(loss_box)
        # TODO ms-thesis-segmentation split here to obtain the two different modules
        if self.cfg.MODEL.MASK_ON:
            # TODO ms-thesis-segmentation read in the saved features
            image_name = os.path.splitext(image_name_path)[0]
            #file_name = image_name + '_features.pkl'
            #with open(file_name, 'rb') as f:  # Python 2: open(...) without 'rb'
            #    mask_features = pickle.load(f)
            mask_features = features
            # TODO ms-thesis-segmentation read in the saved detections as long as SVM module is not yet available
            #with open('result.pkl', 'rb') as f:  # Python 2: open(...) without 'rb'
             #   detections = pickle.load(f)

            # Test does not work yet, but somehow they should be converted to cpu?
            #mask_features[0]= mask_features[0].to('cpu')
            #mask_features[1]= mask_features[1].to('cpu')
            #mask_features[2]= mask_features[2].to('cpu')
            #mask_features[3]= mask_features[3].to('cpu')
            #mask_features[4]= mask_features[4].to('cpu')

            if not use_gt:
                # TODO read in Elisas format
                # test_path = os.path.join(os.path.split(os.path.split(os.path.split(image_name)[0])[0])[1],
                #                          os.path.split(os.path.split(image_name)[0])[1], os.path.split(image_name)[1])
                # detection_dir = os.path.join(
                #     os.path.split(os.path.split(os.path.split(os.path.split(image_name)[0])[0])[0])[0], 'detections',
                #     test_path + '.mat')
                # result = scipy.io.loadmat(detection_dir)['bb']
                result = scipy.io.loadmat(detections , squeeze_me=False)['bb']
                # construct BoxList from the .mat file
                labels = []
                scores = []
                bboxes = []
                existsDetection = False
                for i in range(len(result)):
                    # Find all classes which have at least one detection
                    if len(result[i][0]) >0:
                        # loop through all detection of this class
                        if len(result[i][0][0]) >0:

                            # TODO adapt to enable several detections of same object
                            for det in result[i][0]:
                                # Filter out detections with negative scores
                                if(det[4] < 0):
                                    continue
                                existsDetection = True
                                xmin = det[0]
                                ymin = det[1]
                                xmax = det[2]
                                ymax = det[3]
                                # i+1 because labels are one based
                                labels.append(i+1)
                                scores.append(det[4])
                                # convert to 0-based format
                                bboxes.append([xmin -1, ymin - 1, xmax - 1, ymax - 1])

                if not existsDetection:
                    # There is no detection
                    # TODO handle this case!
                    return False, False, False

                width = img_size[0]
                height = img_size[1]

                det_boxlist = BoxList(torch.tensor(bboxes), image_size=(width, height), mode='xyxy')
                det_boxlist.add_field('labels', torch.tensor(labels))
                det_boxlist.add_field('scores', torch.tensor(scores))
            else:
                # use the boxlist saved in detections (contains gt)
                det_boxlist = detections
            # TODO remove these hard coded values
            # width = 1066
            # height = 800
            width = transformed_size[1]
            height = transformed_size[0]
            # rescale boxes
            det_boxlist = det_boxlist.resize((width,height))
            detections = [det_boxlist]


            # TODO ms-thesis-segmentation alternatively read in the simulated SVM output
            # file_name = image_name + '_simulated_SVM_output.mat'
            # SVM_result = scipy.io.loadmat(file_name)
            # # build BBoxList
            # # TODO ms-thesis-segmentation also read in width and height values
            # width = img_size[0]
            # height = img_size[1]
            # # create a BoxList with 3 boxes
            # #bbox = BoxList(torch.from_numpy(SVM_result['BBox']).to('cpu'), image_size=(width, height), mode='xyxy')
            #
            # bbox = BoxList(torch.from_numpy(SVM_result['BBox']), image_size=(width, height), mode='xyxy')
            # # TODO remove these hard coded values
            # width = 1066
            # height = 800
            # # rescale boxes
            # bbox = bbox.resize((width,height))
            #
            # # add labels for each bbox
            # bbox.add_field('labels', torch.from_numpy(SVM_result['labels'][0]))
            # bbox.add_field('scores', torch.from_numpy(SVM_result['scores'][0]))
            # detections = [bbox]

            #mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            #if (
            #    self.training
            #    and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            #):
            #    mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing

            # TODO calculate time here
            start_time = time.time()
            x, detections, loss_mask = self.mask(mask_features, detections, targets)
            curr_time = time.time()
            total_time = (curr_time - start_time)
            log_file = open("log_mask_computation_time.txt", "a")
            log_file.write(str(total_time) + "\n")
            log_file.close()


            #losses.update(loss_mask)

        #if self.cfg.MODEL.KEYPOINT_ON:
            #keypoint_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            #if (
            #    self.training
            #    and self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            #):
            #    keypoint_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            #x, detections, loss_keypoint = self.keypoint(keypoint_features, detections, targets)
            #losses.update(loss_keypoint)
        return x, detections, losses


def build_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards

    #
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    #if not cfg.MODEL.RPN_ONLY:
        #roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
    #if cfg.MODEL.KEYPOINT_ON:
        #roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
