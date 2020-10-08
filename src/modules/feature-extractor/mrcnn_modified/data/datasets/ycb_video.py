import os

import torch
import torch.utils.data
from PIL import Image
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


from maskrcnn_benchmark.structures.bounding_box import BoxList


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
        #print("Image id {} ({}) doesn't have annotations!".format(self.ids[index], target))
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


    def __init__(self, data_dir, image_set, split, use_difficult=False, transforms=None, is_target_task=False, icwt_21_objs=False):

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


        """
        self._imgsetpath = os.path.join(self.root, "ImageSets", self.image_set, self.split + ".txt")

        with open(self._imgsetpath) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]

        if is_target_task is False:
            cls = YCBVideoDataset.CLASSES
        else:
            if icwt_21_objs is False:
                cls = YCBVideoDataset.CLASSES_TARGET_TASK
            else:
                cls = YCBVideoDataset.CLASSES_TARGET_TASK_21_OBJS

        self.class_to_ind = dict(zip(cls, range(len(cls))))

        remove_images_without_annotations = True
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                anno = ET.parse(self._annopath % img_id).getroot()
                anno = self._preprocess_annotation(anno)
                if has_valid_annotation(anno):
                    ids.append(img_id)
                else:
                    print("Image id {} doesn't have annotations!".format(img_id))
            self.ids = ids

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        """
        self.ids = [0, 1] # TODO modify this

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = Image.open(self._imgpath % img_id).convert("RGB")

        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        if len(target) == 0:
            print("Image id {} ({}) doesn't have annotations!".format(self.ids[index], target))

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def __len__(self):
        with open(self._imgsetpath) as f:
            self.ids = f.readlines()
        return len(self.ids)

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]

        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        masks = []
        TO_REMOVE = 1
        
        for obj in target.iter("object"):
            try:
                difficult = int(obj.find("difficult").text) == 1
            except:
                continue
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                bb.find("xmin").text, 
                bb.find("ymin").text, 
                bb.find("xmax").text, 
                bb.find("ymax").text,
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))          
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        return {"height": 480, "width": 640}

        # TODO modify this
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return YCBVideoDataset.CLASSES[class_id]
