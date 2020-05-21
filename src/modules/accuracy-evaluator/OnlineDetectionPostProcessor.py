from maskrcnn_pytorch.benchmark.modeling.roi_heads.box_head.inference import PostProcessor as pp
import torch.nn.functional as F


class OnlineDetectionPostProcessor(pp.PostProcessor):
    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image
        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        class_logits, proposals = x
        class_prob = F.softmax(class_logits, -1)

        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        # concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        # if self.cls_agnostic_bbox_reg:
        #     box_regression = box_regression[:, -4:]
        # proposals = self.box_coder.decode(
        #     box_regression.view(sum(boxes_per_image), -1), concat_boxes
        # )
        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, boxes_per_img, image_shape in zip(
                class_prob, proposals, image_shapes
        ):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = self.filter_results(boxlist, num_classes)
            results.append(boxlist)
        return results