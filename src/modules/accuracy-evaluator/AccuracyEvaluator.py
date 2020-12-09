import sys
import os

from mrcnn_modified.data.datasets.evaluation import evaluate
basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))

import OnlineDetectionPostProcessor_standalone as odp
import yaml

class AccuracyEvaluator():
    def __init__(self, cfg_path, output_folder):
        cfg = yaml.load(open(cfg_path), Loader=yaml.FullLoader)
        self.score_thresh = cfg['EVALUATION']['SCORE_THRESH']
        self.nms = cfg['EVALUATION']['NMS']
        self.detections_per_img = cfg['EVALUATION']['DETECTIONS_PER_IMAGE']
        self.num_classes = cfg['NUM_CLASSES']
        self.output_folder = output_folder

    def evaluate(self, dataset, predictions, cls_agnostic_bbox_reg=True,
                 box_only=False, iou_types=("bbox",), expected_results=(), draw_preds=False,
                 expected_results_sigma_tol=4, is_target_task=False, icwt_21_objs=False):
        print('Evaluating predictions')
        post_processor = odp.OnlineDetectionPostProcessor(score_thresh=self.score_thresh, nms=self.nms,
                                                          detections_per_img=self.detections_per_img,
                                                          cls_agnostic_bbox_reg=cls_agnostic_bbox_reg)

        predictions = post_processor(predictions, self.num_classes)

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
                        output_folder=self.output_folder,
                        **extra_args)
