# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

import argparse
import os

import torch
from mrcnn_modified.config import cfg

from mrcnn_modified.data import make_data_loader

from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer

from mrcnn_modified.modeling.detector.detectors import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer, Checkpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

from mrcnn_modified.engine.inference import inference
import copy
import logging
# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

class AccuracyEvaluatorDetector:
    def __init__(self, cfg_path_target_task=None, local_rank=0):

        self.is_target_task = True
        self.config_file = cfg_path_target_task
        self.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        self.distributed = self.num_gpus > 1
        self.local_rank = local_rank
        self.cfg = cfg.clone()
        self.load_parameters()

        self.falkon_rpn_models = None
        self.regressors_rpn_models = None
        self.stats_rpn = None

        self.falkon_segmentation_models = None
        self.stats_segmentation = None


    def __call__(self, is_train, output_dir=None, train_in_cpu=False, save_features=False, evaluate_segmentation=True, eval_segm_with_gt_bboxes=False, normalize_features_regressors=False, evaluate_segmentation_icwt=False):
        if train_in_cpu:
            self.cfg.MINIBOOTSTRAP.RPN.FEATURES_DEVICE = 'cpu'
            self.cfg.MINIBOOTSTRAP.DETECTOR.FEATURES_DEVICE = 'cpu'
        self.cfg.SAVE_FEATURES_DETECTOR = save_features
        if save_features:
            if output_dir:
                features_path = os.path.join(output_dir, 'features_detector')
                if not os.path.exists(features_path):
                    os.mkdir(features_path)
            else:
                print('Output directory must be specified. Quitting.')
                quit()
        return self.train(is_train, result_dir=output_dir, evaluate_segmentation=evaluate_segmentation, eval_segm_with_gt_bboxes=eval_segm_with_gt_bboxes, normalize_features_regressors=normalize_features_regressors, evaluate_segmentation_icwt=evaluate_segmentation_icwt)

    def load_parameters(self):
        if self.distributed:
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )
            synchronize()
        self.cfg.merge_from_file(self.config_file)
        if self.cfg.MODEL.RPN.RPN_HEAD == 'SingleConvRPNHead_getProposals':
            print('SingleConvRPNHead_getProposals is not correct as RPN head, changed to OnlineRPNHead.')
            self.cfg.MODEL.RPN.RPN_HEAD = 'OnlineRPNHead'
        self.icwt_21_objs = True if str(21) in self.cfg.DATASETS.TRAIN[0] else False
        if self.cfg.OUTPUT_DIR:
            mkdir(self.cfg.OUTPUT_DIR)
        logger = setup_logger("maskrcnn_benchmark", self.cfg.OUTPUT_DIR, get_rank())
        logger.info("Using {} GPUs".format(self.num_gpus))
        logger.info("Collecting env info (might take some time)")
        logger.info("\n" + collect_env_info())
        logger.info("Loaded configuration file {}".format(self.config_file))
        with open(self.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
        logger.info("Running with config:\n{}".format(self.cfg))


    def train(self, is_train, result_dir=False, evaluate_segmentation=True, eval_segm_with_gt_bboxes=False, normalize_features_regressors=False, evaluate_segmentation_icwt=False):

        model = build_detection_model(self.cfg)
        device = torch.device(self.cfg.MODEL.DEVICE)
        model.to(device)

        optimizer = make_optimizer(self.cfg, model)
        scheduler = make_lr_scheduler(self.cfg, optimizer)

        # Initialize mixed-precision training
        use_mixed_precision = self.cfg.DTYPE == "float16"
        amp_opt_level = 'O1' if use_mixed_precision else 'O0'
        model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

        if self.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_rank], output_device=self.local_rank,
                # this should be removed if we update BatchNorm stats
                broadcast_buffers=False,
            )

        arguments = {}
        arguments["iteration"] = 0

        output_dir = self.cfg.OUTPUT_DIR

        save_to_disk = get_rank() == 0
        checkpointer = DetectronCheckpointer(
            self.cfg, model, None, None, output_dir, save_to_disk
        )

        if self.cfg.MODEL.WEIGHT.startswith('/') or 'catalog' in self.cfg.MODEL.WEIGHT:
            model_path = self.cfg.MODEL.WEIGHT
        else:
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir, 'Data', 'pretrained_feature_extractors', self.cfg.MODEL.WEIGHT))

        extra_checkpoint_data = checkpointer.load(model_path)

        if self.falkon_rpn_models is not None:
            model.rpn.head.classifiers = self.falkon_rpn_models            
        if self.regressors_rpn_models is not None:
            model.rpn.head.regressors = self.regressors_rpn_models
        if self.stats_rpn is not None:
            model.rpn.head.stats = self.stats_rpn

        if self.falkon_detector_models is not None:
            model.roi_heads.box.predictor.classifiers = self.falkon_detector_models
        if self.regressors_detector_models is not None:
            model.roi_heads.box.predictor.regressors = self.regressors_detector_models
        if self.stats_detector is not None:
            model.roi_heads.box.predictor.stats = self.stats_detector

        model.roi_heads.box.predictor.normalize_features_regressors = normalize_features_regressors

        if self.falkon_segmentation_models is not None:
            model.roi_heads.mask.predictor.classifiers = self.falkon_segmentation_models
        if self.stats_segmentation is not None:
            model.roi_heads.mask.predictor.stats = self.stats_segmentation

        if self.distributed:
            model = model.module

        iou_types = ("bbox",)
        torch.cuda.empty_cache()  # TODO check if it helps

        output_folders = [None]
        if is_train:
            dataset_names = ['train']
        else:
            dataset_names = ['test']

        if self.cfg.OUTPUT_DIR:
            for idx, dataset_name in enumerate(dataset_names):
                output_folder = os.path.join(self.cfg.OUTPUT_DIR, dataset_name)
                mkdir(output_folder)
                output_folders[idx] = output_folder

        data_loaders = make_data_loader(self.cfg, is_train=is_train, is_distributed=self.distributed, is_final_test=True, is_target_task=self.is_target_task, icwt_21_objs=self.icwt_21_objs)

        for output_folder, dataset_name, data_loader in zip(output_folders, dataset_names, data_loaders):
            feat_extraction_time = inference(self.cfg,
                                             model,
                                             data_loader,
                                             dataset_name=dataset_name,
                                             iou_types=iou_types,
                                             box_only=False if self.cfg.MODEL.RETINANET_ON else self.cfg.MODEL.RPN_ONLY,
                                             device=cfg.MODEL.DEVICE,
                                             is_target_task=self.is_target_task,
                                             icwt_21_objs=self.icwt_21_objs,
                                             is_train = is_train,
                                             result_dir=result_dir,
                                             evaluate_segmentation=evaluate_segmentation,
                                             eval_segm_with_gt_bboxes=eval_segm_with_gt_bboxes,
                                             evaluate_segmentation_icwt=evaluate_segmentation_icwt
                                            )

            if result_dir and is_train:
                with open(os.path.join(result_dir, "result.txt"), "a") as fid:
                    fid.write("Detector's feature extraction time: {}min:{}s \n".format(int(feat_extraction_time/60), round(feat_extraction_time%60)))

            synchronize()
        logger = logging.getLogger("maskrcnn_benchmark")
        logger.handlers = []
        return
