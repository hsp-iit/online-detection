# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from mrcnn_modified.modeling.rpn.rpn_getProposals_RPN import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util

import numpy as np
import os

import copy


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

class OnlineRPNFeatureExtractorDemo(object):
    CATEGORIES_RPN = [
        "0",
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
        '10',
        '11',
        '12',
        '13',
        '14',
    ]


    def __init__(
        self,
        cfg,
        models_dir,
        max_training_images,
    ):
        self.cfg = cfg.clone()
        self.cfg.NUM_IMAGES = max_training_images
        self.model = build_detection_model(self.cfg)


        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        #if self.cfg.MODEL.WEIGHT.startswith('/') or 'catalog' in self.cfg.MODEL.WEIGHT:
        #    model_path = self.cfg.MODEL.WEIGHT
        #else:
        #    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir, 'Data', 'pretrained_feature_extractors', self.cfg.MODEL.WEIGHT))

        #model_pretrained = torch.load(self.cfg.MODEL.WEIGHT)
        #model_pretrained_copy = copy.deepcopy(checkpointer.model)
        #for key in model_pretrained_copy['model'].keys():
        #    if key.startswith('roi'):
        #        del model_pretrained['model'][key]
        #checkpointer._load_model(model_pretrained)
        
        self.transforms = self.build_transform(cfg)


        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.CATEGORIES = self.CATEGORIES_RPN

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

    def compute_features(self, original_image, gt_bbox_boxlist, gt_classes_list):
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
            #predictions = self.model(image_list, img_size=img_size)[1]
            predictions = self.model(image_list, gt_bbox=gt_bbox_boxlist, gt_label=gt_labels_torch, img_size=img_sizes, gt_labels_list=gt_classes_list, is_train=True, result_dir='')
        """
        if not type(predictions) is list:
            predictions = [predictions]
        predictions = [o.to(self.cpu_device) for o in predictions]

        try:
            # always single image is passed at a time
            prediction = predictions[0]
        except:
            return None

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
        return prediction
        """

