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
from mrcnn_modified.engine.trainer import do_train
#from maskrcnn_benchmark.modeling.detector import build_detection_model
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
    def __init__(self, cfg_path_feature_task=None, local_rank=0, use_backbone_features=False):
        self.is_target_task = False
        self.config_file = cfg_path_feature_task
        self.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        self.distributed = self.num_gpus > 1
        self.local_rank = local_rank
        self.cfg = cfg.clone()
        self.load_parameters()
        self.use_backbone_features = use_backbone_features


    def __call__(self, output_dir=None, fine_tune_last_layers=False, fine_tune_rpn=False, training_seconds=None):
        self.train(output_dir=output_dir, fine_tune_last_layers=fine_tune_last_layers, fine_tune_rpn=fine_tune_rpn, training_seconds=training_seconds)

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

    def train(self, output_dir=None, fine_tune_last_layers=False, fine_tune_rpn=False, training_seconds=None):
        if output_dir is not None:
            self.cfg.OUTPUT_DIR = output_dir
        if self.use_backbone_features:
            from mrcnn_modified.modeling.detector.detectors_train_from_backbone_features import build_detection_model
            # Build the complete model for testing, to avoid saving features for the val/test set
            from maskrcnn_benchmark.modeling.detector import build_detection_model as build_detection_model_val
        else:
            from maskrcnn_benchmark.modeling.detector import build_detection_model
            # If features are not loaded from features, training and validation models are the same
            model_val = None
            checkpointer_val = None
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
        if self.use_backbone_features:
            cfg_val = self.cfg.clone()
            #cfg_val.MODEL.ROI_BOX_HEAD.NUM_CLASSES = cfg_val.MINIBOOTSTRAP.DETECTOR.NUM_CLASSES + 1
            model_val = build_detection_model_val(cfg_val)

            model_val.to(device)
            checkpointer_val = DetectronCheckpointer(
                cfg_val, model_val, None, None, output_dir, save_to_disk
            )

        if self.cfg.MODEL.WEIGHT.startswith('/') or 'catalog' in self.cfg.MODEL.WEIGHT:
            model_path = self.cfg.MODEL.WEIGHT
        else:
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir, 'Data', 'pretrained_feature_extractors', self.cfg.MODEL.WEIGHT))

        extra_checkpoint_data = checkpointer.load(model_path)

        # Initialize the final layer with the correct number of classes
        checkpointer.model.roi_heads.box.predictor.cls_score = torch.nn.Linear(in_features=checkpointer.model.roi_heads.box.predictor.cls_score.in_features, out_features=self.cfg.MINIBOOTSTRAP.DETECTOR.NUM_CLASSES+1, bias=True)
        checkpointer.model.roi_heads.box.predictor.bbox_pred = torch.nn.Linear(in_features=checkpointer.model.roi_heads.box.predictor.cls_score.in_features, out_features=(self.cfg.MINIBOOTSTRAP.DETECTOR.NUM_CLASSES+1)*4, bias=True)
        if hasattr(checkpointer.model.roi_heads, 'mask'):
            checkpointer.model.roi_heads.mask.predictor.mask_fcn_logits = torch.nn.Conv2d(in_channels=checkpointer.model.roi_heads.mask.predictor.mask_fcn_logits.in_channels, out_channels=self.cfg.MINIBOOTSTRAP.DETECTOR.NUM_CLASSES+1, kernel_size=(1, 1), stride=(1, 1))
        checkpointer.model.to(device)


        if fine_tune_last_layers:
            checkpointer.model.roi_heads.box.predictor.cls_score = torch.nn.Linear(in_features=checkpointer.model.roi_heads.box.predictor.cls_score.in_features, out_features=self.cfg.MINIBOOTSTRAP.DETECTOR.NUM_CLASSES+1, bias=True)
            checkpointer.model.roi_heads.box.predictor.bbox_pred = torch.nn.Linear(in_features=checkpointer.model.roi_heads.box.predictor.cls_score.in_features, out_features=(self.cfg.MINIBOOTSTRAP.DETECTOR.NUM_CLASSES+1)*4, bias=True)
            if hasattr(checkpointer.model.roi_heads, 'mask'):
                checkpointer.model.roi_heads.mask.predictor.mask_fcn_logits = torch.nn.Conv2d(in_channels=checkpointer.model.roi_heads.mask.predictor.mask_fcn_logits.in_channels, out_channels=self.cfg.MINIBOOTSTRAP.DETECTOR.NUM_CLASSES+1, kernel_size=(1, 1), stride=(1, 1))
            # Freeze backbone layers if the model has the backbone
            if hasattr(checkpointer.model, 'backbone'):
                for elem in checkpointer.model.backbone.parameters():
                    elem.requires_grad = False
            if not fine_tune_rpn:
                # Freeze RPN layers
                for elem in checkpointer.model.rpn.parameters():
                    elem.requires_grad = False
            else:
                for elem in checkpointer.model.rpn.head.conv.parameters():
                    elem.requires_grad = False
                checkpointer.model.rpn.head.cls_logits = torch.nn.Conv2d(in_channels=checkpointer.model.rpn.head.cls_logits.in_channels, out_channels=checkpointer.model.rpn.head.cls_logits.out_channels, kernel_size=(1, 1), stride=(1, 1))
                checkpointer.model.rpn.head.bbox_pred = torch.nn.Conv2d(in_channels=checkpointer.model.rpn.head.bbox_pred.in_channels, out_channels=checkpointer.model.rpn.head.bbox_pred.out_channels, kernel_size=(1, 1), stride=(1, 1))
            # Freeze roi_heads layers with the exception of the predictor ones
            for elem in checkpointer.model.roi_heads.box.feature_extractor.parameters():
                elem.requires_grad = False
            for elem in checkpointer.model.roi_heads.box.predictor.parameters():
                elem.requires_grad = True
            if hasattr(checkpointer.model.roi_heads, 'mask'):
                for elem in checkpointer.model.roi_heads.mask.predictor.parameters():
                    elem.requires_grad = False
                for elem in checkpointer.model.roi_heads.mask.predictor.mask_fcn_logits.parameters():
                    elem.requires_grad = True
            checkpointer.model.to(device)

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

        if model_val:
            # Load weights of all the net, otherwise the backbone is excluded if the weights will be copied only later from the trained model
            extra_checkpoint_data_val = checkpointer_val.load(model_path)
            # Substitute final layers to be sure that the number of outputs is the same
            checkpointer_val.model.roi_heads.box.predictor.cls_score = checkpointer.model.roi_heads.box.predictor.cls_score
            checkpointer_val.model.roi_heads.box.predictor.bbox_pred = checkpointer.model.roi_heads.box.predictor.bbox_pred
            checkpointer_val.model.roi_heads.mask.predictor.mask_fcn_logits = checkpointer.model.roi_heads.mask.predictor.mask_fcn_logits

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
            checkpointer.optimizer,
            checkpointer.scheduler,
            checkpointer,
            device,
            checkpoint_period,
            test_period,
            arguments,
            is_target_task=self.is_target_task,
            training_seconds=training_seconds,
            model_val=model_val,
            checkpointer_val=checkpointer_val
        )

        logger = logging.getLogger("maskrcnn_benchmark")
        logger.handlers=[]
