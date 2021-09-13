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
from torchvision import transforms as T
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

from .icubworld import iCubWorldDataset



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


class iCubWorldDatasetFromFeat(iCubWorldDataset):

    CLASSES = (
        "__background__",
        "cellphone1", "cellphone2", "cellphone3", "cellphone4", "cellphone5", "cellphone6", "cellphone7", "cellphone8", "cellphone9", "cellphone10",
        "mouse1", "mouse2", "mouse3", "mouse4", "mouse5", "mouse6", "mouse7", "mouse8", "mouse9", "mouse10",
        "perfume1", "perfume2", "perfume3", "perfume4", "perfume5", "perfume6", "perfume7", "perfume8", "perfume9", "perfume10",
        "remote1", "remote2", "remote3", "remote4", "remote5", "remote6", "remote7", "remote8", "remote9", "remote10",
        "soapdispenser1", "soapdispenser2", "soapdispenser3", "soapdispenser4", "soapdispenser5", "soapdispenser6", "soapdispenser7", "soapdispenser8", "soapdispenser9", "soapdispenser10",
        "sunglasses1", "sunglasses2", "sunglasses3", "sunglasses4", "sunglasses5", "sunglasses6", "sunglasses7", "sunglasses8", "sunglasses9", "sunglasses10",
        "glass1", "glass2", "glass3", "glass4", "glass5", "glass6", "glass7", "glass8", "glass9", "glass10",
        "hairbrush1", "hairbrush2", "hairbrush3", "hairbrush4", "hairbrush5", "hairbrush6", "hairbrush7", "hairbrush8", "hairbrush9", "hairbrush10",
        "ovenglove1", "ovenglove2", "ovenglove3", "ovenglove4", "ovenglove5", "ovenglove6", "ovenglove7", "ovenglove8", "ovenglove9", "ovenglove10",
        "squeezer1", "squeezer2", "squeezer3", "squeezer4", "squeezer5", "squeezer6", "squeezer7", "squeezer8", "squeezer9", "squeezer10"
    )
    CLASSES_TARGET_TASK = (
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
    )
    CLASSES_TARGET_TASK_21_OBJS = (
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
    )
    CLASSES_YCBV_IN_HAND = (
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
        "024_bowl",
        "025_mug",
        "035_power_drill",
        "036_wood_block",
        "037_scissors",
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


    def __init__(self, data_dir, image_set, split, use_difficult=False, transforms=None, is_target_task=False, icwt_21_objs=False, remove_images_without_annotations=True):

        self.root = data_dir
        self.image_set = image_set
        self.split = split

        self.keep_difficult = use_difficult
        self.transforms = transforms

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "Images", "%s.jpg")
        self._maskpath = os.path.join(self.root, "Masks", "%s.png")
        self.compute_masks = False
                
        self._imgsetpath = os.path.join(self.root, "ImageSets", self.image_set, self.split + ".txt")

        with open(self._imgsetpath) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]

        if 'ycbv' in data_dir:
            cls = iCubWorldDataset.CLASSES_YCBV_IN_HAND
            self.compute_masks = True
        elif 'HO3D' in data_dir:
            cls = iCubWorldDataset.CLASSES_HO3D
            self._imgpath = self._imgpath.replace('.jpg', '.png')
            self.compute_masks = True
        else:
            if is_target_task is False:
                cls = iCubWorldDataset.CLASSES
            else:
                if icwt_21_objs is False:
                    cls = iCubWorldDataset.CLASSES_TARGET_TASK
                else:
                    cls = iCubWorldDataset.CLASSES_TARGET_TASK_21_OBJS

        self.class_to_ind = dict(zip(cls, range(len(cls))))

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

    def __getitem__(self, index):

        filename_path = self._imgpath % (self.ids[index])
        filename_path = filename_path.replace('Images', 'Features').replace('.png', '.pth')
        img = torch.load(filename_path)[0]

        target_path = filename_path.replace('Features', 'Targets')
        target = torch.load(target_path)

        return img, target, index

    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index, evaluate_segmentation=False):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        if evaluate_segmentation:
            mask_path = (self._maskpath % img_id)
            masks = SegmentationMask(T.ToTensor()(Image.open(mask_path)), (width, height), mode="mask")


        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        if evaluate_segmentation:
            target.add_field("masks", masks)

        return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        masks = []
        if 'HO3D' or 'ycbv' in self.root:
            TO_REMOVE = 0
        else:
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
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id, is_target_task=False, icwt_21_objs=False):
        if 'ycbv' in self.root:
            return iCubWorldDataset.CLASSES_YCBV_IN_HAND[class_id]
        elif 'HO3D' in self.root:
            return iCubWorldDataset.CLASSES_HO3D[class_id]
        else:
            if is_target_task is False:
                return iCubWorldDataset.CLASSES[class_id]
            else:
                if icwt_21_objs is False:
                    return iCubWorldDataset.CLASSES_TARGET_TASK[class_id]
                else:
                    return iCubWorldDataset.CLASSES_TARGET_TASK_21_OBJS[class_id]
