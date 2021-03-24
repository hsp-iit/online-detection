import logging

from .icw_eval import do_icw_evaluation


def icw_evaluation(dataset, predictions, output_folder, box_only, iou_types, expected_results, expected_results_sigma_tol, draw_preds, is_target_task=False, icwt_21_objs=False, iou_thresholds=(0.5,), use_07_metric=True, evaluate_segmentation=False):

    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("icw evaluation doesn't support box_only, ignored.")
    logger.info("performing icw evaluation, ignored iou_types.")

    return do_icw_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        draw_preds=draw_preds,
        logger=logger,
        is_target_task=is_target_task,
        icwt_21_objs=icwt_21_objs,
        iou_thresholds=iou_thresholds,
        use_07_metric=use_07_metric,
        evaluate_segmentation=evaluate_segmentation
    )
