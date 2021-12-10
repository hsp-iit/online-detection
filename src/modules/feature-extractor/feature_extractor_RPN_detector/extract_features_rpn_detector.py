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

from mrcnn_modified.modeling.detector.detectors_getProposals import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer, Checkpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

from mrcnn_modified.engine.feature_proposal_extractor import inference
import copy
import logging
# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

import time

class FeatureExtractorRPNDetector:
    def __init__(self, cfg_path_target_task=None, local_rank=0):

        self.is_target_task = True
        self.config_file = cfg_path_target_task
        self.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        self.distributed = self.num_gpus > 1
        self.local_rank = local_rank
        self.cfg = cfg.clone()
        self.load_parameters()
        self.start_of_feature_extraction_time = None
        self.end_of_feature_extraction_time = None


    def __call__(self, is_train, output_dir=None, train_in_cpu=False, save_features=False, extract_features_segmentation=False, use_only_gt_positives_detection=True, cfg_options={}):
        if train_in_cpu:
            self.cfg.MINIBOOTSTRAP.RPN.FEATURES_DEVICE = 'cpu'
            self.cfg.MINIBOOTSTRAP.DETECTOR.FEATURES_DEVICE = 'cpu'
        self.cfg.SAVE_FEATURES_DETECTOR = save_features
        self.cfg.MINIBOOTSTRAP.DETECTOR.EXTRACT_ONLY_GT_POSITIVES = use_only_gt_positives_detection
        self.cfg.SAVE_FEATURES_RPN = save_features

        if 'minibootstrap_iterations' in cfg_options:
            self.cfg.MINIBOOTSTRAP.DETECTOR.ITERATIONS = cfg_options['minibootstrap_iterations']
            self.cfg.MINIBOOTSTRAP.RPN.ITERATIONS = cfg_options['minibootstrap_iterations']

        if save_features:
            if output_dir:
                features_path = os.path.join(output_dir, 'features_RPN')
                if not os.path.exists(features_path):
                    os.mkdir(features_path)
                features_path = os.path.join(output_dir, 'features_detector')
                if not os.path.exists(features_path):
                    os.mkdir(features_path)
                if extract_features_segmentation:
                    features_path_segm = os.path.join(output_dir, 'features_segmentation')
                    if not os.path.exists(features_path_segm):
                        os.mkdir(features_path_segm)
            else:
                print('Output directory must be specified. Quitting.')
                quit()

        return self.train(is_train, result_dir=output_dir, extract_features_segmentation=extract_features_segmentation, use_only_gt_positives_detection=use_only_gt_positives_detection)

    def load_parameters(self):
        if self.distributed:
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )
            synchronize()
        self.cfg.merge_from_file(self.config_file)
        if self.cfg.MODEL.RPN.RPN_HEAD == 'OnlineRPNHead':
            print('OnlineRPNHead is not correct as RPN head, changed to SingleConvRPNHead_getProposals.')
            self.cfg.MODEL.RPN.RPN_HEAD = 'SingleConvRPNHead_getProposals'
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


    def train(self, is_train, result_dir=False, extract_features_segmentation=False, use_only_gt_positives_detection=True):
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
            self.cfg, model, optimizer, scheduler, output_dir, save_to_disk
        )

        if self.cfg.MODEL.WEIGHT.startswith('/') or 'catalog' in self.cfg.MODEL.WEIGHT:
            model_path = self.cfg.MODEL.WEIGHT
        else:
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir, 'Data', 'pretrained_feature_extractors', self.cfg.MODEL.WEIGHT))

        extra_checkpoint_data = checkpointer.load(model_path)


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

        torch.cuda.synchronize()
        self.start_of_feature_extraction_time = time.time()

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
                                             extract_features_segmentation=extract_features_segmentation,
                                            )

            if result_dir and is_train:
                with open(os.path.join(result_dir, "result.txt"), "a") as fid:
                    fid.write("Detector's feature extraction time: {}min:{}s \n".format(int(feat_extraction_time/60), round(feat_extraction_time%60)))

            torch.cuda.synchronize()
            self.end_of_feature_extraction_time = time.time()

            if is_train:
                logger = logging.getLogger("maskrcnn_benchmark")
                logger.handlers=[]

                if self.cfg.FINE_TUNING_OPTIONS.TRAIN_FROM_FEATURES:
                    return

                if self.cfg.SAVE_FEATURES_DETECTOR:
                    # Save features still not saved
                    for clss in range(len(model.roi_heads.box.negatives)):
                        for batch in range(len(model.roi_heads.box.negatives[clss])):
                            if model.roi_heads.box.negatives[clss][batch].size()[0] > 0:
                                path_to_save = os.path.join(result_dir, 'features_detector', 'negatives_cl_{}_batch_{}'.format(clss, batch))
                                torch.save(model.roi_heads.box.negatives[clss][batch], path_to_save)
                        if use_only_gt_positives_detection:
                            # If a class does not have positive examples, save an empty tensor
                            if model.roi_heads.box.positives[clss][0].size()[0] == 0 and len(model.roi_heads.box.positives[clss]) == 1:
                                path_to_save = os.path.join(result_dir, 'features_detector', 'positives_cl_{}_batch_{}'.format(clss, 0))
                                torch.save(torch.empty((0, model.roi_heads.box.feat_size), device=model.roi_heads.box.negatives[clss][0].device), path_to_save)
                            else:
                                for batch in range(len(model.roi_heads.box.positives[clss])):
                                    if model.roi_heads.box.positives[clss][batch].size()[0] > 0:
                                        path_to_save = os.path.join(result_dir, 'features_detector', 'positives_cl_{}_batch_{}'.format(clss, batch))
                                        torch.save(model.roi_heads.box.positives[clss][batch], path_to_save)

                        if extract_features_segmentation:
                            # If a class does not have positive examples, save an empty tensor
                            if model.roi_heads.mask.positives[clss][0].size()[0] == 0 and len(
                                    model.roi_heads.mask.positives[clss]) == 1:
                                path_to_save = os.path.join(result_dir, 'features_segmentation',
                                                            'positives_cl_{}_batch_{}'.format(clss, 0))
                                torch.save(torch.empty((0, model.roi_heads.mask.feat_size),
                                                       device=model.roi_heads.mask.negatives[clss][0].device),
                                           path_to_save)
                            else:
                                for batch in range(len(model.roi_heads.mask.positives[clss])):
                                    if model.roi_heads.mask.positives[clss][batch].size()[0] > 0:
                                        path_to_save = os.path.join(result_dir, 'features_segmentation',
                                                                    'positives_cl_{}_batch_{}'.format(clss, batch))
                                        torch.save(model.roi_heads.mask.positives[clss][batch], path_to_save)

                            # If a class does not have positive examples, save an empty tensor
                            if model.roi_heads.mask.negatives[clss][0].size()[0] == 0 and len(
                                    model.roi_heads.mask.negatives[clss]) == 1:
                                path_to_save = os.path.join(result_dir, 'features_segmentation',
                                                            'negatives_cl_{}_batch_{}'.format(clss, 0))
                                torch.save(torch.empty((0, model.roi_heads.mask.feat_size),
                                                       device=model.roi_heads.mask.negatives[clss][0].device),
                                           path_to_save)
                            else:
                                for batch in range(len(model.roi_heads.mask.negatives[clss])):
                                    if model.roi_heads.mask.negatives[clss][batch].size()[0] > 0:
                                        path_to_save = os.path.join(result_dir, 'features_segmentation',
                                                                    'negatives_cl_{}_batch_{}'.format(clss, batch))
                                        torch.save(model.roi_heads.mask.negatives[clss][batch], path_to_save)

                    for i in range(len(model.roi_heads.box.X)):
                        if model.roi_heads.box.X[i].size()[0] > 0:
                            path_to_save = os.path.join(result_dir, 'features_detector', 'reg_x_batch_{}'.format(i))
                            torch.save(model.roi_heads.box.X[i], path_to_save)

                            path_to_save = os.path.join(result_dir, 'features_detector', 'reg_c_batch_{}'.format(i))
                            torch.save(model.roi_heads.box.C[i], path_to_save)

                            path_to_save = os.path.join(result_dir, 'features_detector', 'reg_y_batch_{}'.format(i))
                            torch.save(model.roi_heads.box.Y[i], path_to_save)
                    if not self.cfg.SAVE_FEATURES_RPN:
                        if extract_features_segmentation:
                            if use_only_gt_positives_detection:
                                return None, None, None, None, None, None, None, None
                            else:
                                return None, None, None, None, None, None, None, None

                        else:
                            if use_only_gt_positives_detection:
                                return None, None, None, None, None, None
                            else:
                                return None, None, None, None, None, None

                if self.cfg.SAVE_FEATURES_RPN:
                    # Save features still not saved
                    for clss in model.rpn.anchors_ids:
                        # Save negatives batches
                        for batch in range(len(model.rpn.negatives[clss])):
                            if model.rpn.negatives[clss][batch].size()[0] > 0:
                                path_to_save = os.path.join(result_dir, 'features_RPN',
                                                            'negatives_cl_{}_batch_{}'.format(clss, batch))
                                torch.save(model.rpn.negatives[clss][batch], path_to_save)
                        # If a class does not have positive examples, save an empty tensor
                        if model.rpn.positives[clss][0].size()[0] == 0 and len(model.rpn.positives[clss]) == 1:
                            path_to_save = os.path.join(result_dir, 'features_RPN',
                                                        'positives_cl_{}_batch_{}'.format(clss, 0))
                            torch.save(
                                torch.empty((0, model.rpn.feat_size), device=model.rpn.negatives[clss][0].device),
                                path_to_save)
                        else:
                            for batch in range(len(model.rpn.positives[clss])):
                                if model.rpn.positives[clss][batch].size()[0] > 0:
                                    path_to_save = os.path.join(result_dir, 'features_RPN',
                                                                'positives_cl_{}_batch_{}'.format(clss, batch))
                                    torch.save(model.rpn.positives[clss][batch], path_to_save)

                    for i in range(len(model.rpn.X)):
                        if model.rpn.X[i].size()[0] > 0:
                            path_to_save = os.path.join(result_dir, 'features_RPN', 'reg_x_batch_{}'.format(i))
                            torch.save(model.rpn.X[i], path_to_save)

                            path_to_save = os.path.join(result_dir, 'features_RPN', 'reg_c_batch_{}'.format(i))
                            torch.save(model.rpn.C[i], path_to_save)

                            path_to_save = os.path.join(result_dir, 'features_RPN', 'reg_y_batch_{}'.format(i))
                            torch.save(model.rpn.Y[i], path_to_save)

                    if extract_features_segmentation:
                        if use_only_gt_positives_detection:
                            return None, None, None, None, None, None, None, None
                        else:
                            return None, None, None, None, None, None, None, None

                    else:
                        if use_only_gt_positives_detection:
                            return None, None, None, None, None, None
                        else:
                            return None, None, None, None, None, None

                if not self.cfg.SAVE_FEATURES_RPN and not self.cfg.SAVE_FEATURES_DETECTOR:

                    COXY_rpn = {'C': torch.cat(model.rpn.C),
                                'O': model.rpn.O,
                                'X': torch.cat(model.rpn.X),
                                'Y': torch.cat(model.rpn.Y)
                               }
                    for i in range(self.cfg.MINIBOOTSTRAP.RPN.NUM_CLASSES):
                        model.rpn.positives[i] = torch.cat(model.rpn.positives[i])
                        if self.cfg.MINIBOOTSTRAP.RPN.SHUFFLE_NEGATIVES:
                            total_negatives_i = torch.cat(model.rpn.negatives[i])
                            shuffled_ids = torch.randperm(len(total_negatives_i))
                            model.rpn.negatives[i] = []
                            for j in range(self.cfg.MINIBOOTSTRAP.RPN.ITERATIONS):
                                start_j_index = min(j * self.cfg.MINIBOOTSTRAP.RPN.BATCH_SIZE, len(shuffled_ids))
                                end_j_index = min((j + 1) * self.cfg.MINIBOOTSTRAP.RPN.BATCH_SIZE, len(shuffled_ids))
                                model.rpn.negatives[i].append(total_negatives_i[shuffled_ids[start_j_index:end_j_index]])

                    COXY = {'C': torch.cat(model.roi_heads.box.C),
                            'O': model.roi_heads.box.O,
                            'X': torch.cat(model.roi_heads.box.X),
                            'Y': torch.cat(model.roi_heads.box.Y)
                            }
                    for i in range(self.cfg.MINIBOOTSTRAP.DETECTOR.NUM_CLASSES):
                        if use_only_gt_positives_detection:
                            model.roi_heads.box.positives[i] = torch.cat(model.roi_heads.box.positives[i])
                        if self.cfg.MINIBOOTSTRAP.DETECTOR.SHUFFLE_NEGATIVES:
                            total_negatives_i = torch.cat(model.roi_heads.box.negatives[i])
                            shuffled_ids = torch.randperm(len(total_negatives_i))
                            model.roi_heads.box.negatives[i] = []
                            for j in range(self.cfg.MINIBOOTSTRAP.DETECTOR.ITERATIONS):
                                start_j_index = min(j * self.cfg.MINIBOOTSTRAP.DETECTOR.BATCH_SIZE, len(shuffled_ids))
                                end_j_index = min((j + 1) * self.cfg.MINIBOOTSTRAP.DETECTOR.BATCH_SIZE, len(shuffled_ids))
                                model.roi_heads.box.negatives[i].append(total_negatives_i[shuffled_ids[start_j_index:end_j_index]])
                        if extract_features_segmentation:
                            if self.cfg.SEGMENTATION.FEATURES_DEVICE == 'cpu':
                                model.roi_heads.mask.negatives[i][len(model.roi_heads.mask.negatives[i])-1] = model.roi_heads.mask.negatives[i][len(model.roi_heads.mask.negatives[i])-1].to('cpu')
                            model.roi_heads.mask.negatives[i] = torch.cat(model.roi_heads.mask.negatives[i])
                            if self.cfg.SEGMENTATION.FEATURES_DEVICE == 'cpu':
                                model.roi_heads.mask.positives[i][len(model.roi_heads.mask.positives[i])-1] = model.roi_heads.mask.positives[i][len(model.roi_heads.mask.positives[i])-1].to('cpu')
                            model.roi_heads.mask.positives[i] = torch.cat(model.roi_heads.mask.positives[i])
                    if extract_features_segmentation:
                        if use_only_gt_positives_detection:
                            return model.rpn.negatives, model.rpn.positives, COXY_rpn, model.roi_heads.box.negatives, model.roi_heads.box.positives, COXY, model.roi_heads.mask.negatives, model.roi_heads.mask.positives
                        else:
                            return model.rpn.negatives, model.rpn.positives, COXY_rpn, model.roi_heads.box.negatives, None, COXY, model.roi_heads.mask.negatives, model.roi_heads.mask.positives

                    else:
                        if use_only_gt_positives_detection:
                            return model.rpn.negatives, model.rpn.positives, COXY_rpn, model.roi_heads.box.negatives, model.roi_heads.box.positives, COXY
                        else:
                            return model.rpn.negatives, model.rpn.positives, COXY_rpn, model.roi_heads.box.negatives, None, COXY

            else:
                logger = logging.getLogger("maskrcnn_benchmark")
                logger.handlers=[]
                return model.roi_heads.box.test_boxes
