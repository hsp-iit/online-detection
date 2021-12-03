import os

import torch
import torch.utils.data
from PIL import Image
import sys
import json

from torchvision import transforms as T

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

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


class YCBVideoDataset(torch.utils.data.Dataset):

    CLASSES = (
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
    )

    CLASSES_HO3D = (
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
    )

    CLASSES_NOT_IN_HO3D = (
        "__background__",
        "002_master_chef_can",
        "005_tomato_soup_can",
        "007_tuna_fish_can",
        "008_pudding_box",
        "009_gelatin_box",
        "019_pitcher_base",
        "024_bowl",
        "036_wood_block",
        "040_large_marker",
        "051_large_clamp",
        "052_extra_large_clamp",
        "061_foam_brick"
    )

    def __init__(self, data_dir, image_set, split, use_difficult=False, transforms=None, is_target_task=False, icwt_21_objs=False, remove_images_without_annotations=True, ycbv_classes_not_in_ho3d=False):

        self.root = data_dir
        self.image_set = image_set
        self.split = split

        self.keep_difficult = use_difficult
        self.transforms = transforms

        if 'pbr' in self.root:
            self._imgpath = os.path.join(self.root, "%s", "rgb", "%s.jpg")
        else:
            self._imgpath = os.path.join(self.root, "%s", "rgb", "%s.png")

        self._maskpath = os.path.join(self.root, "%s", "mask_visib", "%s.png")

        self._scene_gt_info_path = os.path.join(self.root, "%s", "scene_gt_info.json")
        self._scene_gt_path = os.path.join(self.root, "%s", "scene_gt.json")

        self._imgsetpath = os.path.join(self.root, self.split + ".txt")

        imset = open(self._imgsetpath, "r")
        folder_list = []
        folder_list_int = []
        for line in imset.readlines():
            folder_num = line.split()[0]
            if folder_num not in folder_list:
                folder_list.append(folder_num)
                folder_list_int.append(int(folder_num))
        folder_list = sorted(folder_list)

        self.scene_gts = [None] * (max(folder_list_int)+1)
        self.scene_gt_infos = [None] * (max(folder_list_int)+1)
        for i in range(len(folder_list)):
            f = open(self._scene_gt_path % folder_list[i])
            scene_gt = json.load(f)
            f.close()
            self.scene_gts[int(folder_list[i])] = scene_gt
            f = open(self._scene_gt_info_path % folder_list[i])
            scene_gt_info = json.load(f)
            f.close()
            self.scene_gt_infos[int(folder_list[i])] = scene_gt_info

        with open(self._imgsetpath) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]

        self.ycbv_classes_not_in_ho3d = ycbv_classes_not_in_ho3d
        if self.ycbv_classes_not_in_ho3d:
            imgset_path = self._imgsetpath
            ids = []
            for index in range(len(self.ids)):
                imset = open(imgset_path, "r")
                img_path = imset.readlines()[index].strip('\n').split()
                scene_gt = self.scene_gts[int(img_path[0])]
                scene_gt_info = self.scene_gt_infos[int(img_path[0])]

                removed = False
                for j in range(len(scene_gt[str(int(img_path[1]))])):
                    bbox = scene_gt_info[str(int(img_path[1]))][j]["bbox_visib"]
                    if YCBVideoDataset.CLASSES[scene_gt[str(int(img_path[1]))][j]["obj_id"]] not in YCBVideoDataset.CLASSES_HO3D and bbox != [-1, -1, -1, -1] and bbox[2] != 0 and bbox[3] != 0:
                        ids.append(self.ids[index])
                        imset.close()
                        removed = True
                        break
                if not removed:
                    print("Image {} not used".format(self.ids[index]))

            self.ids = ids



    def __getitem__(self, index):

        filename_path = self._imgpath % (self.ids[index].strip('\n').split()[0], self.ids[index].strip('\n').split()[1])
        img = Image.open(filename_path).convert("RGB")

        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        if len(target) == 0:
            print("Image id {} ({}) doesn't have annotations!".format(self.ids[index], target))

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def __len__(self):
        return len(self.ids)

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

        target = BoxList(torch.tensor(gt_bboxes_list), (width, height), mode="xyxy")
        masks = SegmentationMask(torch.cat(masks), (width, height), mode="mask")
        target.add_field("labels", torch.tensor(gt_labels))
        target.add_field("masks", masks)
        target.add_field("difficult", torch.tensor(difficult_boxes))

        return target

    def get_img_info(self, index):
        return {"height": 480, "width": 640}

    def map_class_id_to_class_name(self, class_id):
        if not self.ycbv_classes_not_in_ho3d:
            return YCBVideoDataset.CLASSES[class_id]
        else:
            return YCBVideoDataset.CLASSES_NOT_IN_HO3D[class_id]
