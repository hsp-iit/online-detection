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

from mrcnn_modified.modeling.rpn.rpn_getProposals_RPN import build_detection_model
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

class FeatureExtractorRPN:
    def __init__(self, cfg_path_RPN=None, local_rank=0):

        self.is_target_task = True
        self.config_file = cfg_path_RPN
        self.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        self.distributed = self.num_gpus > 1
        self.local_rank = local_rank
        self.cfg = cfg.clone()
        self.load_parameters()
        self.start_of_feature_extraction_time = None

    def __call__(self, is_train, output_dir=None, train_in_cpu=False, save_features=False, cfg_options={}):
        if train_in_cpu:
            self.cfg.MINIBOOTSTRAP.RPN.FEATURES_DEVICE = 'cpu'
        self.cfg.SAVE_FEATURES_RPN = save_features
        if save_features:
            if output_dir:
                features_path = os.path.join(output_dir, 'features_RPN')
                if not os.path.exists(features_path):
                    os.mkdir(features_path)
            else:
                print('Output directory must be specified. Quitting.')
                quit()
        if 'minibootstrap_iterations' in cfg_options:
            self.cfg.MINIBOOTSTRAP.RPN.ITERATIONS = cfg_options['minibootstrap_iterations']

        return self.train(is_train, result_dir=output_dir)


    def load_parameters(self):
        if self.distributed:
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )
            synchronize()
        self.cfg.merge_from_file(self.config_file)
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


    def train(self, is_train, result_dir=None):
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
        # Load rpn
        if self.cfg.MODEL.WEIGHT.startswith('/') or 'catalog' in self.cfg.MODEL.WEIGHT:
            model_path = self.cfg.MODEL.WEIGHT
        else:
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir, 'Data', 'pretrained_feature_extractors', self.cfg.MODEL.WEIGHT))

        checkpointer = DetectronCheckpointer(cfg, model, save_dir=result_dir)
        _ = checkpointer.load(model_path)

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

            torch.cuda.synchronize()
            self.start_of_feature_extraction_time = time.time()

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
                                            )

            if result_dir and is_train:
                with open(os.path.join(result_dir, "result.txt"), "a") as fid:
                    fid.write("RPN's feature extraction time: {}min:{}s \n".format(int(feat_extraction_time/60), round(feat_extraction_time%60)))

            synchronize()
        logger = logging.getLogger("maskrcnn_benchmark")
        logger.handlers=[]
        if self.cfg.SAVE_FEATURES_RPN:
            # Save features still not saved
            for clss in model.rpn.anchors_ids:
                # Save negatives batches
                for batch in range(len(model.rpn.negatives[clss])):
                    if model.rpn.negatives[clss][batch].size()[0] > 0:
                        path_to_save = os.path.join(result_dir, 'features_RPN', 'negatives_cl_{}_batch_{}'.format(clss, batch))
                        torch.save(model.rpn.negatives[clss][batch], path_to_save)
                # If a class does not have positive examples, save an empty tensor
                if model.rpn.positives[clss][0].size()[0] == 0 and len(model.rpn.positives[clss]) == 1:
                    path_to_save = os.path.join(result_dir, 'features_RPN', 'positives_cl_{}_batch_{}'.format(clss, 0))
                    torch.save(torch.empty((0, model.rpn.feat_size), device=model.rpn.negatives[clss][0].device), path_to_save)
                else:
                    for batch in range(len(model.rpn.positives[clss])):
                        if model.rpn.positives[clss][batch].size()[0] > 0:
                            path_to_save = os.path.join(result_dir, 'features_RPN', 'positives_cl_{}_batch_{}'.format(clss, batch))
                            torch.save(model.rpn.positives[clss][batch], path_to_save)

            for i in range(len(model.rpn.X)):
                if model.rpn.X[i].size()[0] > 0:
                    path_to_save = os.path.join(result_dir, 'features_RPN', 'reg_x_batch_{}'.format(i))
                    torch.save(model.rpn.X[i], path_to_save)

                    path_to_save = os.path.join(result_dir, 'features_RPN', 'reg_c_batch_{}'.format(i))
                    torch.save(model.rpn.C[i], path_to_save)

                    path_to_save = os.path.join(result_dir, 'features_RPN', 'reg_y_batch_{}'.format(i))
                    torch.save(model.rpn.Y[i], path_to_save)
            return
        else:
            COXY = {'C': torch.cat(model.rpn.C),
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

            return model.rpn.negatives, model.rpn.positives, COXY

