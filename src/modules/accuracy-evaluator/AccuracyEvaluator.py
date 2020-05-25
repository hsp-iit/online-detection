import sys
import os

from maskrcnn_pytorch.benchmark.data.datasets.evaluation import evaluate
basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))

import OnlineDetectionPostProcessor as odp


class AccuracyEvaluator():
    def __init__(self, score_thresh=0.05, nms=0.3, detections_per_img=100, cls_agnostic_bbox_reg=True):
        self.post_processor = odp.OnlineDetectionPostProcessor(score_thresh=score_thresh, nms=nms,
                                                              detections_per_img=detections_per_img,
                                                              cls_agnostic_bbox_reg=cls_agnostic_bbox_reg)

    def evaluate(self, dataset, predictions, opts,
                 box_only=False, iou_types=("bbox",), expected_results=(), draw_preds=False,
                 expected_results_sigma_tol=4, is_target_task=False, icwt_21_objs=False):
        print('Evaluating predictions')

        predictions = self.post_processor(predictions, opts['num_classes'])

        extra_args = dict(
            box_only=box_only,
            iou_types=iou_types,
            expected_results=expected_results,
            expected_results_sigma_tol=expected_results_sigma_tol,
            draw_preds=draw_preds,
            is_target_task=is_target_task,
            icwt_21_objs=icwt_21_objs
        )

        return evaluate(dataset=dataset,
                        predictions=predictions,
                        output_folder=opts['output_folder'],
                        **extra_args)
