# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.comm import synchronize
from maskrcnn_benchmark.utils.timer import Timer, get_time_str

import torch
from PIL import Image
import numpy as np

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

import glob
# To parse the annotation .xml files
import xml.etree.ElementTree as ET

from torchvision import transforms as T
from maskrcnn_benchmark.structures.image_list import to_image_list

OBJECTNAME_TO_ID = {
    "__background__":0,
        "flower2":1, "flower5":2, "flower7":3,
        "mug1":4, "mug3":5, "mug4":6,
        "wallet6":7, "wallet7":8, "wallet10":9,
        "sodabottle2":10, "sodabottle3":11, "sodabottle4":12,
        "book4":13, "book6":14, "book9":15,
        "ringbinder4":16, "ringbinder5":17, "ringbinder6":18,
        "bodylotion2":19, "bodylotion5":20, "bodylotion8":21,
        "sprayer6":22, "sprayer8":23, "sprayer9":24,
        "pencilcase3":25, "pencilcase5":26, "pencilcase6":27,
        "hairclip2":28, "hairclip6":29, "hairclip8":30,
}

OBJECTNAME_TO_ID_21 = {
    "__background__":0,
        "sodabottle3":1, "sodabottle4":2,
        "mug1":3, "mug3":4, "mug4":5,
        "pencilcase5":6, "pencilcase3":7,
        "ringbinder4":8, "ringbinder5":9,
        "wallet6":10,
        "flower7":11, "flower5":12, "flower2":13,
        "book6":14, "book9":15,
        "hairclip2":16, "hairclip8":17, "hairclip6":18,
        "sprayer6":19, "sprayer8":20, "sprayer9":21,
}

OBJECTNAME_TO_ID_YCBV_IN_HAND = {
        "__background__":0,
        "002_master_chef_can":1,
        "003_cracker_box":2,
        "004_sugar_box":3,
        "005_tomato_soup_can":4,
        "006_mustard_bottle":5,
        "007_tuna_fish_can":6,
        "008_pudding_box":7,
        "009_gelatin_box":8,
        "010_potted_meat_can":9,
        "011_banana":10,
        "019_pitcher_base":11,
        "024_bowl":12,
        "025_mug":13,
        "035_power_drill":14,
        "036_wood_block":15,
        "037_scissors":16,
        "051_large_clamp":17,
        "052_extra_large_clamp":18,
        "061_foam_brick":19
}

OBJECTNAME_TO_ID_HO3D = {
    "__background__":0,
    "003_cracker_box":1,
    "004_sugar_box":2,
    "006_mustard_bottle":3,
    "010_potted_meat_can":4,
    "011_banana":5,
    "021_bleach_cleanser":6,
    "025_mug":7,
    "035_power_drill":8,
    "037_scissors":9,
}

def build_transform(cfg):
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

def compute_gts_icwt(dataset, i, icwt_21_objs = None):
    img_dir = dataset._imgpath
    anno_dir = dataset._annopath
    mask_dir = dataset._maskpath
    img_path = dataset.ids[i]

    filename_path = img_dir % img_path
    print(filename_path)
    img_RGB = Image.open(filename_path)
    # get image size such that later the boxes can be resized to the correct size
    width, height = img_RGB.size
    img_sizes = [width, height]
    # convert to BGR format
    try:
        image = np.array(img_RGB)[:, :, [2, 1, 0]]
    except:
        image = np.array(img_RGB.convert('RGB'))[:, :, [2, 1, 0]]
    mask_path = (mask_dir % img_path)

    mask = None
    if os.path.exists(mask_path):
        mask = T.ToTensor()(Image.open(mask_path)).to('cuda')
    # Read in annotation file
    anno_file = anno_dir % img_path
    tree = ET.parse(anno_file, ET.XMLParser(encoding='utf-8'))
    root = tree.getroot()
    # Read label
    gt_labels = []
    gt_bboxes_list = []
    masks = []
    for object in root.findall('object'):
        try:
            name = object.find('name').text
        except:
            continue

        if not icwt_21_objs:
            if 'ycbv' in anno_dir:
                gt_label = OBJECTNAME_TO_ID_YCBV_IN_HAND[name]
            elif 'HO3D' in anno_dir:
                gt_label = OBJECTNAME_TO_ID_HO3D[name]
            else:
                gt_label = OBJECTNAME_TO_ID[name]
        else:
            gt_label = OBJECTNAME_TO_ID_21[name]

        gt_labels.append(gt_label)

        xmin = object.find('bndbox').find('xmin').text
        ymin = object.find('bndbox').find('ymin').text
        xmax = object.find('bndbox').find('xmax').text
        ymax = object.find('bndbox').find('ymax').text

        if 'HO3D' or 'ycbv' not in anno_dir:
            gt_bboxes_list.append([float(xmin) - 1, float(ymin) - 1, float(xmax) - 1, float(ymax) - 1])
        else:
            gt_bboxes_list.append([float(xmin), float(ymin), float(xmax), float(ymax)])

        # Please note that that masks gts works only with the modified version of iCWT in which there is only an object per image
        # In the case that on-line segmentation will be necessary on a different extension of iCWT with possibly more than an object per image,
        # this function will be extended according to annotations' format
        if mask is not None:
            masks.append(mask)

    return image, gt_bboxes_list, masks, gt_labels, img_sizes, filename_path

def compute_gts_ycbv(dataset, i, extract_features_segmentation):

    img_dir = dataset._imgpath
    mask_dir = dataset._maskpath
    img_path = dataset.ids[i].split()
    filename_path = img_dir % (img_path[0], img_path[1])

    scene_gt = dataset.scene_gts[int(img_path[0])]
    scene_gt_info = dataset.scene_gt_infos[int(img_path[0])]

    print(filename_path)
    img_RGB = Image.open(filename_path)
    # get image size such that later the boxes can be resized to the correct size
    width, height = img_RGB.size
    img_sizes = [width, height]
    # convert to BGR format
    try:
        image = np.array(img_RGB)[:, :, [2, 1, 0]]
    except:
        image = np.array(img_RGB.convert('RGB'))[:, :, [2, 1, 0]]

    masks_paths = sorted(glob.glob(mask_dir%(img_path[0], img_path[1]+'*')))

    gt_labels = []
    gt_bboxes_list = []
    masks = []

    for j in range(len(masks_paths)):
        bbox = scene_gt_info[str(int(img_path[1]))][j]["bbox_visib"]
        if bbox == [-1, -1, -1, -1] or bbox[2] == 0 or bbox[3] == 0:
            continue

        obj_id = scene_gt[str(int(img_path[1]))][j]["obj_id"]
        # Manage the self.ycbv_classes_not_in_ho3d case
        if dataset.ycbv_classes_not_in_ho3d:
            # Do not consider gts belonging to classes in ho3d
            if dataset.CLASSES[obj_id] in dataset.CLASSES_HO3D:
                continue
            else:
                obj_id = dataset.CLASSES_NOT_IN_HO3D.index(dataset.CLASSES[obj_id])
        gt_bboxes_list.append([bbox[0], bbox[1], bbox[0] + bbox[2] - 1, bbox[1] + bbox[3] - 1])
        gt_labels.append(obj_id)
        if extract_features_segmentation:
            masks.append(T.ToTensor()(Image.open(masks_paths[j])).to('cuda'))

    return image, gt_bboxes_list, masks, gt_labels, img_sizes, filename_path

def extract_feature_proposals(cfg, dataset, model, transforms, icwt_21_objs=False, compute_average_recall_RPN = False, is_train = True, result_dir = None, extract_features_segmentation=False):

    model.eval()
    num_img = len(dataset.ids)

    # Set the number of images that will be used to set minibootstrap parameters
    if hasattr(model, 'rpn'):
        model.rpn.cfg.NUM_IMAGES = num_img
    if hasattr(model, 'roi_heads'):
        model.roi_heads.box.cfg.NUM_IMAGES = num_img

    if compute_average_recall_RPN:
        average_recall_RPN = 0

    for i in range(num_img):
        if type(dataset).__name__ is 'iCubWorldDataset':
            image, gt_bboxes_list, masks, gt_labels, img_sizes, img_name = compute_gts_icwt(dataset, i, icwt_21_objs)
        elif type(dataset).__name__ is 'YCBVideoDataset':
            image, gt_bboxes_list, masks, gt_labels, img_sizes, img_name = compute_gts_ycbv(dataset, i, extract_features_segmentation=extract_features_segmentation)

        # Save list of boxes as tensor
        gt_bbox_tensor = torch.tensor(gt_bboxes_list, device="cuda")
        gt_labels_torch = torch.tensor(gt_labels, device="cuda", dtype=torch.uint8).reshape((len(gt_labels),1))

        if len(masks) > 0:
            mask_lists = SegmentationMask(torch.cat(masks), img_sizes, mode='mask')

        # create box list containing the ground truth bounding boxes
        try:
            gt_bbox_boxlist = BoxList(gt_bbox_tensor, image_size=img_sizes, mode='xyxy')
            try:
                gt_bbox_boxlist.add_field("masks", mask_lists)
            except:
                pass
        except:
            gt_bbox_boxlist = BoxList(torch.empty((0,4), device="cuda"), image_size=img_sizes, mode='xyxy')

        gt_bbox_boxlist.add_field("labels", torch.tensor(gt_labels))
            
        # apply pre-processing to image
        image = transforms(image)
        # convert to an ImageList
        image_list = to_image_list(image, 1)
        image_list = image_list.to("cuda")
        # compute predictions
        with torch.no_grad():
            AR = model(image_list, gt_bbox=gt_bbox_boxlist, gt_label=gt_labels_torch, img_size=img_sizes, compute_average_recall_RPN=compute_average_recall_RPN, gt_labels_list=gt_labels, is_train=is_train, result_dir=result_dir, extract_features_segmentation=extract_features_segmentation, img_name=img_name)
            if compute_average_recall_RPN:
                average_recall_RPN += AR

    if compute_average_recall_RPN:
        return average_recall_RPN / num_img
    else:
        return None

def inference(
        cfg,
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        draw_preds=False,
        is_target_task=False,
        icwt_21_objs=False,
        compute_average_recall_RPN=False,
        is_train = True,
        result_dir=None,
        extract_features_segmentation=False,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    AR = extract_feature_proposals(cfg, dataset, model, build_transform(cfg), icwt_21_objs, compute_average_recall_RPN= not is_train, is_train=is_train, result_dir=result_dir, extract_features_segmentation=extract_features_segmentation)
    print('Average Recall (AR):', AR)

    if result_dir and not is_train:
        with open(os.path.join(result_dir, "result.txt"), "a") as fid:
            fid.write('Average Recall (AR): {} \n \n'.format(AR))

    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    return total_time
