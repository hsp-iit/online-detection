import os
from PIL import Image
import torch
import numpy as np
import random

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


class IVOSDataset(object):

    def __init__(self, data_dir, image_set, split, transforms=None,):
        self.root = data_dir
        self.image_set = image_set
        self.classes = (
                       "__background__", "apple1", "apple2", "banana", "bottle1", "bottle2", "bottle3", "bottle4",
                       "bowl1", "bowl2", "bowl3", "bowl4", "bowl5", "mug1", "mug2", "mug3", "mug4", "mug5", "mug6", "mug7", "yogurt"
                       )
        self.split = split
        self._imgsetpath = os.path.join(self.root, self.image_set + ".txt")
        #self.transforms = ("Rotation", "Scale", "Translation")
        with open(self._imgsetpath) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        """
        ids_to_remove = []
        for i in range(len(self.ids)):
            img_path = os.path.join(self.root, self.ids[i])
            image = Image.open(img_path).convert("RGB")
            mask_path = img_path.replace("Images", "Masks")
            try:
                mask = Image.open(mask_path).convert("1")
            except:
                ids_to_remove.append(i)
        """
        ids_to_remove = [5004, 5005, 5018, 5055, 5079, 5081, 5087, 5088, 5092, 5123, 5143, 5149, 5157, 5166, 5184, 5189, 5828, 5831, 5929, 5964, 6124, 7869, 7917, 7948, 7969, 9847, 10413, 10420, 10482, 10571, 10800]
        for i in reversed(ids_to_remove):
            del self.ids[i]
        random.seed(1)
        ids_sample = random.sample(self.ids, round(len(self.ids)*10/100))
        if split == "train":
            for elem in ids_sample:
                self.ids.remove(elem)
            #print(len(self.ids))
        elif split == "test":
            self.ids = ids_sample
        self.transforms = transforms

    def __getitem__(self, idx):
        # load the image as a PIL Image
        img_path = os.path.join(self.root, self.ids[idx])
        image = Image.open(img_path).convert("RGB")
        mask_path = img_path.replace("Images", "Masks")
        mask = Image.open(mask_path).convert("1")
        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        #print(mask_path)
        #print(image.height, image.width)
        bbox = mask.getbbox()
        #print(bbox)
        boxes = [[bbox[0], bbox[1], bbox[2], bbox[3]]]
        # and labels
        for elem in self.classes:
            if elem in img_path:
                labels = torch.tensor([self.classes.index(elem)])
        #labels = torch.tensor([10, 20])
        masks = torch.as_tensor([np.array(mask)], dtype=torch.uint8)
        #masks = [mask]
        #print(image.size)
        #print(masks.shape)
        #print(idx)
        #print(img_path)
        #print(mask_path)
        #print(mask.mode)
        #if "mug1" in img_path:
        #    print(masks)
        masks = SegmentationMask(masks, image.size, mode = "mask")
        #print(mask_path)
        # create a BoxList from the boxes
        boxlist = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)
        boxlist.add_field("masks", masks)

        if self.transforms:
            image, boxlist = self.transforms(image, boxlist)
        boxlist.add_field("difficult", torch.tensor([[0]]))
        # return the image, the boxlist and the idx in your dataset
        return image, boxlist, idx

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        img_path = os.path.join(self.root, self.ids[idx])
        img = Image.open(img_path)#.convert("RGB")
        return {"height": img.height, "width": img.width}

    def __len__(self):
        return len(self.ids)
    

    def map_class_id_to_class_name(self, class_id):
        return self.classes[class_id]
