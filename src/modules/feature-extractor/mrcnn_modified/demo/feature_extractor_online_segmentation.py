# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from mrcnn_modified.modeling.detector.detectors_getProposals import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util

import numpy as np
import os

class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        return image

class OnlineSegmentationFeatureExtractorDemo(object):
    CATEGORIES_iCWT_TT_21 = [
        "__background__",
        'sodabottle3', 'sodabottle4',
        'mug1', 'mug3', 'mug4',
        'pencilcase5', 'pencilcase3',
        'ringbinder4', 'ringbinder5',
        'wallet6',
        'flower7', 'flower5', 'flower2',
        'book6', 'book9',
        'hairclip2', 'hairclip8', 'hairclip6',
        'sprayer6', 'sprayer8', 'sprayer9'
    ]
    CATEGORIES_iCWT_TT = [
        "__background__",
        "flower2", "flower5", "flower7",
        "mug1", "mug3", "mug4",
        "wallet6", "wallet7", "wallet10",
        "sodabottle2", "sodabottle3", "sodabottle4",
        "book4", "book6", "book9",
        "ringbinder4", "ringbinder5", "ringbinder6",
        "bodylotion2", "bodylotion5", "bodylotion8",
        "sprayer6", "sprayer8", "sprayer9",
        "pencilcase3", "pencilcase5", "pencilcase6",
        "hairclip2", "hairclip6", "hairclip8"

    ]
    CATEGORIES_YCBV = [
        "__background__",
        "002_master_chef_can",
        "003_cracker_box",
        "004_sugar_box",
        "005_tomato_soup_can",
        "006_mustard_bottle",
        "007_tuna_fish_can",
        "008_pudding_box",
        "009_gelatin_box",
        "010_potted_meat_can",
        "011_banana",
        "019_pitcher_base",
        "021_bleach_cleanser",
        "024_bowl",
        "025_mug",
        "035_power_drill",
        "036_wood_block",
        "037_scissors",
        "040_large_marker",
        "051_large_clamp",
        "052_extra_large_clamp",
        "061_foam_brick"
    ]

    def __init__(
        self,
        cfg,
        confidence_threshold,
        models_dir,
        dataset,
        max_training_images,
        num_classes_trained_together
    ):
        self.cfg = cfg.clone()
        self.cfg.MINIBOOTSTRAP.DETECTOR.NUM_CLASSES = num_classes_trained_together
        self.cfg.NUM_IMAGES = max_training_images
        if self.cfg.MODEL.RPN.RPN_HEAD == 'SingleConvRPNHead_getProposals':
            print('SingleConvRPNHead_getProposals is not correct as RPN head, changed to OnlineRPNHead.')
            self.cfg.MODEL.RPN.RPN_HEAD = 'OnlineRPNHead'
        self.model = build_detection_model(self.cfg)

        try:
            self.model.rpn.head.classifiers = torch.load(os.path.join(models_dir, 'classifier_rpn'))
            self.model.rpn.head.regressors = torch.load(os.path.join(models_dir, 'regressor_rpn'))
            self.model.rpn.head.stats = torch.load(os.path.join(models_dir, 'stats_rpn'))
        except:
            pass


        self.model.eval()
        self.device = torch.device(self.cfg.MODEL.DEVICE)
        self.model.to(self.device)

        save_dir = self.cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(self.cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(self.cfg.MODEL.WEIGHT)
        
        self.transforms = self.build_transform(self.cfg)

        self.masker = Masker(threshold=0.5, padding=1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold

        if dataset == 'iCWT_TT':
            self.CATEGORIES = self.CATEGORIES_iCWT_TT
        elif dataset == 'iCWT_TT_TABLETOP':
            self.CATEGORIES = self.CATEGORIES_iCWT_TT_21
        elif dataset == 'ycbv':
            self.CATEGORIES = self.CATEGORIES_YCBV
        else:
            self.CATEGORIES = None

    def build_transform(self, cfg):
        """
        Creates a basic transformation that was used to train the models
        """
        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(cfg.INPUT.MIN_SIZE_TEST),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def compute_features(self, original_image, gt_bbox_boxlist, gt_classes_list, extract_features_segmentation=False):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        img_sizes = [original_image.shape[1], original_image.shape[0]]
        gt_labels_torch = torch.tensor(gt_classes_list, device="cuda", dtype=torch.uint8).reshape((len(gt_classes_list), 1))
        with torch.no_grad():
            predictions = self.model(image_list, gt_bbox=gt_bbox_boxlist, gt_label=gt_labels_torch, img_size=img_sizes, gt_labels_list=gt_classes_list, is_train=True, result_dir='',  extract_features_segmentation=extract_features_segmentation)

    def update_model(self, models_rpn=None):
        if models_rpn:
            self.model.rpn.head.classifiers = models_rpn['classifiers']
            self.model.rpn.head.regressors = models_rpn['regressors']
            self.model.rpn.head.stats = models_rpn['stats']

