# A modification version from chainercv repository.
# (See https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_voc.py)
from __future__ import division

import os
from collections import defaultdict
import numpy as np
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from mrcnn_modified.modeling.roi_heads.mask_head.inference import Masker
from py_od_utils import mask_iou

import cv2
import torch


def select_top_predictions(predictions, confidence_threshold):
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
    keep = torch.nonzero(scores > confidence_threshold).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    _, idx = scores.sort(0, descending=True)
    return predictions[idx]


def compute_colors_for_labels(labels):
    """
    Simple function that adds fixed colors depending on the class
    """
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

    colors = labels[:, None].type(torch.LongTensor) * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors


def overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = predictions.get_field("labels")
    boxes = predictions.bbox

    colors = compute_colors_for_labels(labels).tolist()

    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), 1
        )

    return image


def overlay_class_names(image, predictions, CATEGORIES):
    """
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    """
    scores = predictions.get_field("scores").tolist()
    labels = predictions.get_field("labels").type(torch.LongTensor).tolist()
    labels = [CATEGORIES[i] for i in labels]
    boxes = predictions.bbox

    template = "{}: {:.2f}"
    for box, score, label in zip(boxes, scores, labels):
        x, y = box[:2]
        s = template.format(label, score)
        cv2.putText(
            image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
        )

    return image


def overlay_labels(image, gt, CATEGORIES):
    """
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    """

    labels = gt.get_field("labels").type(torch.LongTensor).tolist()
    labels = [CATEGORIES[i] for i in labels]
    boxes = gt.bbox

    template = "{}"
    for box, label in zip(boxes, labels):
        x, y = box[:2]
        s = template.format(label)
        cv2.putText(
            image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
        )

    return image


def draw_preds_icw(dataset, image_id, pred, gt, output_folder):

    result = cv2.imread(dataset._imgpath % dataset.ids[image_id])

    top_pred = select_top_predictions(pred, 0.7)

    result = overlay_boxes(result, top_pred)
    result = overlay_boxes(result, gt)
    result = overlay_class_names(result, top_pred, dataset.CLASSES)
    result = overlay_labels(result, gt, dataset.CLASSES)

    output_path = os.path.join(output_folder, "preds", dataset.ids[image_id] + ".jpg")
    output_dir = os.path.dirname(output_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(output_path, result)

    #cv2.imshow('result', result)
    #cv2.waitKey(10)
    #cv2.destroyAllWindows()

    return result


def do_icw_evaluation(dataset, predictions, output_folder, draw_preds, logger, iou_thresholds=(0.5,), use_07_metric=True, is_target_task=False, icwt_21_objs=False, evaluate_segmentation=False):

    pred_boxlists = []
    gt_boxlists = []
    for image_id, prediction in enumerate(predictions):

        img_info = dataset.get_img_info(image_id)

        if len(prediction) == 0:
            logger.info("No predictions for image: {}".format(image_id))
            #continue

        image_width = img_info["width"]
        image_height = img_info["height"]
        if isinstance(prediction, list):
            if len(prediction) == 1:
                prediction = prediction[0]
        prediction = prediction.resize((image_width, image_height))
        pred_boxlists.append(prediction)

        gt_boxlist = dataset.get_groundtruth(image_id, evaluate_segmentation=evaluate_segmentation)
        gt_boxlists.append(gt_boxlist)

        if draw_preds:
            draw_preds_icw(dataset, image_id, prediction, gt_boxlist, output_folder)

    for iou_thresh in iou_thresholds:

        result = eval_detection_icw(
            pred_boxlists=pred_boxlists,
            gt_boxlists=gt_boxlists,
            iou_thresh=iou_thresh,
            use_07_metric=use_07_metric,
        )

        result_str = "Detection mAP{}: {:.4f}\n\n".format(int(iou_thresh*100), result["map"])
        for i, ap in enumerate(result["ap"]):
            if i == 0:  # skip background
                continue
            result_str += "{:<26}: {:.4f}\n".format(
                dataset.map_class_id_to_class_name(i, is_target_task = is_target_task, icwt_21_objs=icwt_21_objs), ap
            )
        result_str += "\n"

        logger.info(result_str)

        if output_folder:
            with open(os.path.join(output_folder, "result.txt"), "a") as fid:
                fid.write(result_str)

        if evaluate_segmentation:
            result = eval_segmentation_ycbv(
                pred_boxlists=pred_boxlists,
                gt_boxlists=gt_boxlists,
                iou_thresh=iou_thresh,
                use_07_metric=use_07_metric,
            )

            result_str = "Segmentation mAP{}: {:.4f}\n\n".format(int(iou_thresh*100), result["map"])
            for i, ap in enumerate(result["ap"]):
                if i == 0:  # skip background
                    continue
                result_str += "{:<26}: {:.4f}\n".format(
                    dataset.map_class_id_to_class_name(i), ap
                )
            result_str += "\n"

            logger.info(result_str)

            if output_folder:
                with open(os.path.join(output_folder, "result.txt"), "a") as fid:
                    fid.write(result_str)

    return result


def eval_detection_icw(pred_boxlists, gt_boxlists, iou_thresh=0.5, use_07_metric=False):
    """Evaluate on voc dataset.
    Args:
        pred_boxlists(list[BoxList]): pred boxlist, has labels and scores fields.
        gt_boxlists(list[BoxList]): ground truth boxlist, has labels field.
        iou_thresh: iou thresh
        use_07_metric: boolean
    Returns:
        dict represents the results
    """
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."
    prec, rec = calc_detection_icw_prec_rec(
        pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, iou_thresh=iou_thresh
    )
    ap = calc_detection_icw_ap(prec, rec, use_07_metric=use_07_metric)
    return {"ap": ap, "map": np.nanmean(ap)}


def calc_detection_icw_prec_rec(gt_boxlists, pred_boxlists, iou_thresh=0.5):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
   """
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):

        pred_bbox = pred_boxlist.bbox.to('cpu').numpy()
        pred_label = pred_boxlist.get_field("labels").to('cpu').numpy()
        pred_score = pred_boxlist.get_field("scores").to('cpu').numpy()
        gt_bbox = gt_boxlist.bbox.to('cpu').numpy()
        gt_label = gt_boxlist.get_field("labels").to('cpu').numpy()
        gt_difficult = gt_boxlist.get_field("difficult").to('cpu').numpy()

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            iou = boxlist_iou(
                BoxList(pred_bbox_l, gt_boxlist.size),
                BoxList(gt_bbox_l, gt_boxlist.size),
            ).numpy()
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    if len(n_pos.keys()) == 0:    #Check if n_pos is empty
        n_fg_class = 1
        print("Returning precision and recall = 0")
        prec = [None]
        rec = [None]
    else:
        n_fg_class = max(n_pos.keys()) + 1
        prec = [None] * n_fg_class
        rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec


def calc_detection_icw_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.
    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def eval_segmentation_ycbv(pred_boxlists, gt_boxlists, iou_thresh=0.5, use_07_metric=False):
    """Evaluate on voc dataset.
    Args:
        pred_boxlists(list[BoxList]): pred boxlist, has labels and scores fields.
        gt_boxlists(list[BoxList]): ground truth boxlist, has labels field.
        iou_thresh: iou thresh
        use_07_metric: boolean
    Returns:
        dict represents the results
    """
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."
    prec, rec = calc_segmentation_voc_prec_rec(
        pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, iou_thresh=iou_thresh
    )
    ap = calc_detection_icw_ap(prec, rec, use_07_metric=use_07_metric)
    return {"ap": ap, "map": np.nanmean(ap)}

def calc_segmentation_voc_prec_rec(gt_boxlists, pred_boxlists, iou_thresh=0.5):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
   """
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    masker = Masker(threshold=0.5, padding=1)
    a = 0
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        a += 1
        pred_label = pred_boxlist.get_field("labels").to('cpu').numpy()
        pred_score = pred_boxlist.get_field("scores").to('cpu').numpy()
        gt_masks = np.array([gt_boxlist.get_field("masks").get_mask_tensor().to('cpu').numpy()])
        if len(gt_masks.shape) == 4:
            gt_masks = np.rint(gt_masks)[0]
        else:
            gt_masks = np.rint(gt_masks)
        gt_masks = np.array(gt_masks, dtype=np.uint8)
        gt_label = gt_boxlist.get_field("labels").to('cpu').numpy()
        gt_difficult = gt_boxlist.get_field("difficult").to('cpu').numpy()

        if pred_boxlist.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = pred_boxlist.get_field("mask")
            # always single image is passed at a time
            pred_masks = masker([masks], [pred_boxlist])[0].numpy().squeeze(1)
        else:
            pred_masks = np.asarray([])

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_masks_l = pred_masks[pred_mask_l]

            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_masks_l = pred_masks_l[order]
            pred_score_l = pred_score_l[order]

            gt_keep_l = gt_label == l
            gt_masks_l = gt_masks[gt_keep_l]

            n_pos[l] += gt_keep_l.sum()
            score[l].extend(pred_score_l)

            if len(pred_masks_l) == 0:
                continue
            if len(gt_masks_l) == 0:
                match[l].extend((0,) * pred_masks_l.shape[0])
                continue

            iou = mask_iou(pred_masks_l, gt_masks_l)
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_masks_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if not selec[gt_idx]:
                        match[l].append(1)
                    else:
                        match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)
    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec
