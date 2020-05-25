# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# https://github.com/hsp-iit/ms-thesis-segmentation/blob/master/maskrcnn_pytorch/tools/train_net.py
r"""
Basic training script for PyTorch
"""
import os
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_pytorch.benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_pytorch.benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
import logging

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

class TrainerFeatureTask:
    def __init__(self, cfg_path_feature_task=None, local_rank=0):
        self.is_target_task = False
        self.config_file = cfg_path_feature_task
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
        self.cfg.freeze()
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
        extra_checkpoint_data = checkpointer.load(self.cfg.MODEL.WEIGHT)
        arguments.update(extra_checkpoint_data)

        data_loader = make_data_loader(
            self.cfg,
            is_train=True,
            is_distributed=self.distributed,
            start_iter=arguments["iteration"],
            is_target_task=self.is_target_task
        )

        test_period = self.cfg.SOLVER.TEST_PERIOD
        if test_period > 0:
            data_loader_val = make_data_loader(self.cfg, is_train=False, is_distributed=self.distributed,
                                               is_target_task=self.is_target_task)
        else:
            data_loader_val = None

        checkpoint_period = self.cfg.SOLVER.CHECKPOINT_PERIOD
        do_train(
            self.cfg,
            model,
            data_loader,
            data_loader_val,
            optimizer,
            scheduler,
            checkpointer,
            device,
            checkpoint_period,
            test_period,
            arguments,
            is_target_task=self.is_target_task
        )

        final_model = torch.load(os.path.join(output_dir, "model_final.pth"))
        torch.save(final_model, "pretrained_feature_extractor.pth")
        print("model_final.pth saved in pretrained_feature_extractor.pth.")

        logger = logging.getLogger("maskrcnn_benchmark")
        logger.handlers=[]
