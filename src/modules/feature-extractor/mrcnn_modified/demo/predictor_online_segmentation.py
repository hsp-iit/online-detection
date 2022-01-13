# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from mrcnn_modified.modeling.detector.detectors import build_detection_model
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

class OnlineSegmentationDemo(object):
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

    CATEGORIES_HO3D = [
        "__background__",
        "003_cracker_box",
        "004_sugar_box",
        "006_mustard_bottle",
        "010_potted_meat_can",
        "011_banana",
        "021_bleach_cleanser",
        "025_mug",
        "035_power_drill",
        "037_scissors",
    ]

    def __init__(
        self,
        cfg,
        confidence_threshold,
        models_dir,
        dataset=None,
        fill_masks=False,
        mask_rcnn_model=False
    ):
        self.cfg = cfg.clone()
        #Set here the confidence threshold to avoid useless computation
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH = confidence_threshold

        if self.cfg.MODEL.RPN.RPN_HEAD == 'SingleConvRPNHead_getProposals':
            print('SingleConvRPNHead_getProposals is not correct as RPN head, changed to OnlineRPNHead.')
            self.cfg.MODEL.RPN.RPN_HEAD = 'OnlineRPNHead'
        self.model = build_detection_model(self.cfg)

        if models_dir:
            try:
                self.model.rpn.head.classifiers = torch.load(os.path.join(models_dir, 'classifier_rpn'))
                self.model.rpn.head.regressors = torch.load(os.path.join(models_dir, 'regressor_rpn'))
                self.model.rpn.head.stats = torch.load(os.path.join(models_dir, 'stats_rpn'))
            except:
                pass

            try:
                self.model.roi_heads.box.predictor.classifiers = torch.load(os.path.join(models_dir, 'classifier_detector'))
                self.model.roi_heads.box.predictor.regressors = torch.load(os.path.join(models_dir, 'regressor_detector'))
                self.model.roi_heads.box.predictor.stats = torch.load(os.path.join(models_dir, 'stats_detector'))
            except:
                pass

            try:
                self.model.roi_heads.mask.predictor.classifiers = torch.load(os.path.join(models_dir, 'classifier_segmentation'))
                self.model.roi_heads.mask.predictor.stats = torch.load(os.path.join(models_dir, 'stats_segmentation'))
            except:
                pass
        elif not mask_rcnn_model:
            self.model.roi_heads.box.predictor.classifiers = []
            self.model.roi_heads.box.predictor.regressors = np.empty((0))
            self.model.roi_heads.box.predictor.stats = None

        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
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
        elif dataset == 'ho3d':
            self.CATEGORIES = self.CATEGORIES_HO3D
        else:
            self.CATEGORIES = None

        self.fill_masks = fill_masks

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

    def run_on_opencv_image(self, image):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions = self.compute_prediction(image)
        if predictions is not None:
            top_predictions = self.select_top_predictions(predictions)
            if len(top_predictions) == 0:
                return image.copy(), None, None
            result = image.copy()
        else:
            return image.copy(), None, None
        result = self.overlay_boxes(result, top_predictions)
        #TODO check the second arguent when using only detection
        if self.cfg.MODEL.MASK_ON:
            result = self.overlay_mask(result, top_predictions)
            result_with_names = self.overlay_class_names(result[0], top_predictions)
            result = (result_with_names[0], result[1], result_with_names[1])
        else:
            result = self.overlay_class_names(result, top_predictions)[0]

        return result

    def compute_prediction(self, original_image):
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
        img_size = [original_image.shape[1], original_image.shape[0]]
        with torch.no_grad():
            predictions = self.model(image_list, img_size=img_size)[1]
        if predictions is None:
            return None
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

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        # Create different colors, otherwise only "green" colors are used
        for i in range(len(colors)):
            np.random.seed(labels[i])
            colors[i] = np.random.permutation(colors[i])
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 1
            )

        return image

    def overlay_mask(self, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")

        colors = self.compute_colors_for_labels(labels).tolist()

        for mask, color in zip(masks, colors):
            thresh = mask[0, :, :, None].astype(np.uint8)
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            image = cv2.drawContours(image, contours, -1, color, 3)

            if self.fill_masks:
                coloredImg = np.zeros(image.shape, image.dtype)
                coloredImg[:, :] = color
                coloredMask = cv2.bitwise_and(coloredImg, coloredImg, mask=mask.astype(np.uint8).squeeze())
                image = cv2.addWeighted(coloredMask, 1, image, 1, 0)

        composite = image

        return composite, masks

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        if self.CATEGORIES is not None:
            labels = [self.CATEGORIES[i] for i in labels]
        else:
            labels = [i for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            s = template.format(label, score)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2
            )
        return image, labels

    def update_model(self, models_rpn=None, models_detection=None, models_segmentation=None, categories=None):
        if models_rpn:
            self.model.rpn.head.classifiers = models_rpn['classifiers']
            self.model.rpn.head.regressors = models_rpn['regressors']
            self.model.rpn.head.stats = models_rpn['stats']
            if hasattr(self.model.rpn.head, 'nystrom_parallel'):
                del self.model.rpn.head.nystrom_parallel
            if hasattr(self.model.rpn.head, 'regressors_parallel'):
                del self.model.rpn.head.regressors_parallel

        if models_detection:
            self.model.roi_heads.box.predictor.classifiers = models_detection['classifiers']
            self.model.roi_heads.box.predictor.regressors = models_detection['regressors']
            self.model.roi_heads.box.predictor.stats = models_detection['stats']
            if hasattr(self.model.roi_heads.box.predictor, 'nystrom_parallel'):
                del self.model.roi_heads.box.predictor.nystrom_parallel
            if hasattr(self.model.roi_heads.box.predictor, 'regressors_parallel'):
                del self.model.roi_heads.box.predictor.regressors_parallel

        if models_segmentation:
            self.model.roi_heads.mask.predictor.classifiers = models_segmentation['classifiers']
            self.model.roi_heads.mask.predictor.stats = models_segmentation['stats']
            if hasattr(self.model.roi_heads.mask.predictor, 'nystrom_parallel'):
                del self.model.roi_heads.mask.predictor.nystrom_parallel

        if categories:
            self.CATEGORIES = categories