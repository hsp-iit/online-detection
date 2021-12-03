import os

import torch
import torch.utils.data
from PIL import Image
import sys
import json

from torchvision import transforms as T

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

from .ycb_video import YCBVideoDataset

import glob

def _has_only_empty_bbox(anno):
    try:
        v = anno["boxes"][:, 2:] <= 1
        return v.numpy().any()
    except:
        return True

def has_valid_annotation(anno):
    height, width = anno["im_info"]

    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False

    target = BoxList(anno["boxes"], (width, height), mode="xyxy")
    target.add_field("labels", anno["labels"])
    target.add_field("difficult", anno["difficult"])
    target = target.clip_to_image(remove_empty=True)
    if len(target) == 0:
        return False

    return True


class YCBVideoDatasetFromFeat(YCBVideoDataset):

    def __getitem__(self, index):

        filename_path = self._imgpath % (self.ids[index].strip('\n').split()[0], self.ids[index].strip('\n').split()[1])
        filename_path = filename_path.replace('rgb', 'feat').replace('.png', '.pth')
        img = torch.load(filename_path)[0]

        target_path = filename_path.replace('feat', 'targets')
        target = torch.load(target_path)

        return img, target, index

    def get_groundtruth(self, index):

        img_dir = self._imgpath
        imgset_path = self._imgsetpath
        mask_dir = self._maskpath

        img_path = self.ids[index].split()

        filename_path = img_dir % (img_path[0], img_path[1])
        scene_gt_path = self._scene_gt_path % img_path[0]

        scene_gt = self.scene_gts[int(img_path[0])]
        scene_gt_info = self.scene_gt_infos[int(img_path[0])]

        img_RGB = Image.open(filename_path)
        # get image size such that later the boxes can be resized to the correct size
        width, height = img_RGB.size
        masks_paths = sorted(glob.glob(mask_dir % (img_path[0], img_path[1] + '*')))

        gt_labels = []
        gt_bboxes_list = []
        masks = []
        difficult_boxes = []

        for j in range(len(masks_paths)):
            bbox = scene_gt_info[str(int(img_path[1]))][j]["bbox_visib"]
            if bbox == [-1, -1, -1, -1] or bbox[2] == 0 or bbox[3] == 0:
                continue
            obj_id = scene_gt[str(int(img_path[1]))][j]["obj_id"]
            #Manage the self.ycbv_classes_not_in_ho3d case
            if self.ycbv_classes_not_in_ho3d:
                # Do not consider gts belonging to classes in ho3d
                if YCBVideoDataset.CLASSES[obj_id] in YCBVideoDataset.CLASSES_HO3D:
                    continue
                else:
                    obj_id = YCBVideoDataset.CLASSES_NOT_IN_HO3D.index(YCBVideoDataset.CLASSES[obj_id])
            gt_bboxes_list.append([bbox[0], bbox[1], bbox[0] + bbox[2] -1, bbox[1] + bbox[3] -1])
            gt_labels.append(obj_id)
            masks.append(T.ToTensor()(Image.open(masks_paths[j])))
            difficult_boxes.append(False)

        #print(torch.tensor(gt_bboxes_list), img_path)
        target = BoxList(torch.tensor(gt_bboxes_list), (width, height), mode="xyxy")
        masks = SegmentationMask(torch.cat(masks), (width, height), mode="mask")
        target.add_field("labels", torch.tensor(gt_labels))
        target.add_field("masks", masks)
        target.add_field("difficult", torch.tensor(difficult_boxes))

        return target