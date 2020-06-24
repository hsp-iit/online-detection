# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# https://github.com/hsp-iit/ms-thesis-segmentation/blob/master/maskrcnn_pytorch/tools/train_net.py
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (do not reorder)

# from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
#from maskrcnn_benchmark.config import cfg
from maskrcnn_pytorch.benchmark.config import cfg

from maskrcnn_pytorch.benchmark.data import make_data_loader
# from maskrcnn_benchmark.data import make_data_loader

from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer

from maskrcnn_pytorch.benchmark.modeling.rpn.rpn_getProposals_RPN import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer, Checkpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

from maskrcnn_pytorch.benchmark.engine.feature_proposal_extractor_new import inference
import copy
import logging
# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

class FeatureExtractorRPN:
    def __init__(self, cfg_path_RPN=None, local_rank=0):

        self.is_target_task = True
        self.config_file = cfg_path_RPN
        self.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        self.distributed = self.num_gpus > 1
        self.local_rank = local_rank
        self.cfg = cfg.clone()
        self.load_parameters()

    def __call__(self):
        self.train()


    def load_parameters(self):
        if self.distributed:
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )
            synchronize()
        self.cfg.merge_from_file(self.config_file)
        self.cfg.OUTPUT_DIR = self.cfg.OUTPUT_DIR % ('R50', '5', 'icwt100', 'icwt30')                   #TODO read these parameters maybe from a config file in the future
        self.cfg.freeze()
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


    def train(self):
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
        model_pretrained = torch.load(self.cfg.MODEL.WEIGHT)
        model_pretrained_copy = copy.deepcopy(model_pretrained)
        for key in model_pretrained_copy['model'].keys():
            if key.startswith('roi'):
                del model_pretrained['model'][key]
        checkpointer._load_model(model_pretrained)

        #TODO add pick final o something similar in checkpointer.load from other training files in the server

        if self.distributed:
            model = model.module

        iou_types = ("bbox",)
        torch.cuda.empty_cache()  # TODO check if it helps

        output_folders = [None] * (len(self.cfg.DATASETS.TRAIN) + len(self.cfg.DATASETS.TEST))
        if len(self.cfg.DATASETS.TRAIN) + len(self.cfg.DATASETS.TEST) == 2:
            dataset_names = ['train_val', 'test']
        else:
            for elem in self.cfg.DATASETS.TRAIN:
                dataset_names.append(elem)
            for elem in self.cfg.DATASETS.TEST:
                dataset_names.append(elem)

        if cfg.OUTPUT_DIR:
            for idx, dataset_name in enumerate(dataset_names):
                output_folder = os.path.join(self.cfg.OUTPUT_DIR, dataset_name)
                mkdir(output_folder)
                output_folders[idx] = output_folder

        data_loaders = make_data_loader(self.cfg, is_train=True, is_distributed=self.distributed,
                                            is_final_test=True, is_target_task=self.is_target_task, icwt_21_objs=self.icwt_21_objs)
        data_loaders_test = make_data_loader(self.cfg, is_train=False, is_distributed=self.distributed,
                                            is_final_test=True, is_target_task=self.is_target_task, icwt_21_objs=self.icwt_21_objs)
        for elem in data_loaders_test:
            data_loaders.append(elem)

        for output_folder, dataset_name, data_loader in zip(output_folders, dataset_names, data_loaders):
            inference(  # TODO change parameters according to the function definition in feature_proposal_extractor
                self.cfg,
                model,
                data_loader,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                output_folder=output_folder,
                is_target_task=self.is_target_task,
                icwt_21_objs=self.icwt_21_objs
            )
            synchronize()

        logger = logging.getLogger("maskrcnn_benchmark")
        logger.handlers=[]
