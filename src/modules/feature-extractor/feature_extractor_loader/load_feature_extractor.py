import torch
from maskrcnn_pytorch.benchmark.config import cfg
import errno
import os

class LoaderFeatureExtractor:
    def __init__(self, cfg_path_target_task=None):

        self.config_file = cfg_path_target_task
        self.cfg = cfg.clone()

    def __call__(self):
        self.load_feature_extractor()

    def load_feature_extractor(self):
        self.cfg.merge_from_file(self.config_file)
        self.cfg.freeze()
        try:
            pretrained_model = torch.load(self.cfg.MODEL.PRETRAINED_DETECTION_WEIGHTS)
            torch.save(pretrained_model, "pretrained_feature_extractor.pth")
            print(self.cfg.MODEL.PRETRAINED_DETECTION_WEIGHTS, "saved in pretrained_feature_extractor.pth.")
        except OSError:
            try:
                torch.load("pretrained_feature_extractor.pth")
            except OSError:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT))
