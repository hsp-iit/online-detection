# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# https://github.com/hsp-iit/ms-thesis-segmentation/blob/master/maskrcnn_pytorch/tools/train_net.py
r"""
Basic training script for PyTorch
"""
import os
import torch
#from maskrcnn_benchmark.config import cfg
from mrcnn_modified.config import cfg
from mrcnn_modified.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from mrcnn_modified.engine.inference_full_mask import inference
from maskrcnn_benchmark.utils.comm import get_world_size, synchronize

import logging

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

class TesterFeatureTask:
    def __init__(self, cfg_path_feature_task=None, local_rank=0):
        self.is_target_task = False
        self.config_file = cfg_path_feature_task
        self.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        self.distributed = self.num_gpus > 1
        self.local_rank = local_rank
        self.cfg = cfg.clone()
        self.load_parameters()

    def __call__(self, output_dir=None, model_to_test=None):
        self.test(output_dir=output_dir, model_to_test=model_to_test)

    def load_parameters(self):
        if self.distributed:
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )
            synchronize()
        self.cfg.merge_from_file(self.config_file)
        #self.cfg.freeze()
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

    def test(self, output_dir=None, model_to_test=None):
        if output_dir is not None:
            self.cfg.OUTPUT_DIR = output_dir
        model = build_detection_model(self.cfg)
        device = torch.device(self.cfg.MODEL.DEVICE)
        model.to(device)

        arguments = {}
        arguments["iteration"] = 0

        output_dir = self.cfg.OUTPUT_DIR

        save_to_disk = get_rank() == 0
        checkpointer = DetectronCheckpointer(
            self.cfg, model, None, None, output_dir, save_to_disk
        )

        if model_to_test is not None:
            self.cfg.MODEL.WEIGHT = model_to_test

        if self.cfg.MODEL.WEIGHT.startswith('/') or 'catalog' in self.cfg.MODEL.WEIGHT:
            model_path = self.cfg.MODEL.WEIGHT
        else:
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir, 'Data', 'pretrained_feature_extractors', self.cfg.MODEL.WEIGHT))

        extra_checkpoint_data = checkpointer.load(model_path, use_latest=False)

        checkpointer.optimizer = make_optimizer(self.cfg, checkpointer.model)
        checkpointer.scheduler = make_lr_scheduler(self.cfg, checkpointer.optimizer)

        # Initialize mixed-precision training
        use_mixed_precision = self.cfg.DTYPE == "float16"
        amp_opt_level = 'O1' if use_mixed_precision else 'O0'
        model, optimizer = amp.initialize(checkpointer.model, checkpointer.optimizer, opt_level=amp_opt_level)

        if self.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_rank], output_device=self.local_rank,
                # this should be removed if we update BatchNorm stats
                broadcast_buffers=False,
            )
        synchronize()
        _ = inference(  # The result can be used for additional logging, e. g. for TensorBoard
            model,
            # The method changes the segmentation mask format in a data loader,
            # so every time a new data loader is created:
            make_data_loader(self.cfg, is_train=False, is_distributed=(get_world_size() > 1), is_target_task=self.is_target_task),
            dataset_name="[Test]",
            iou_types=("bbox",),
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=None,
            is_target_task=self.is_target_task,
        )
        synchronize()

        logger = logging.getLogger("maskrcnn_benchmark")
        logger.handlers=[]
