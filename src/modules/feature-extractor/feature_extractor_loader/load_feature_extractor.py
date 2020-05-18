import torch
from maskrcnn_benchmark.config import cfg
import errno
import os

class LoaderFeatureExtractor:
    def __init__(self, cfg_path_feature_task=None):

        self.config_file = cfg_path_feature_task
        self.cfg = cfg.clone()

    def __call__(self):
        self.load_feature_extractor()

    def load_feature_extractor(self):
        self.cfg.merge_from_file(self.config_file)
        self.cfg.freeze()
        try:
            print(self.cfg.MODEL.WEIGHT)
            torch.load(self.cfg.MODEL.WEIGHT)
        except OSError:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), (self.cfg.MODEL.WEIGHT))
