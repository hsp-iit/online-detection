from RegionRefinerAbstract import RegionRefinerAbstract

from region_refiner_trainer import RegionRefinerTrainer
from region_predictor import RegionPredictor
import yaml


class RegionRefiner(RegionRefinerAbstract):
    def __init__(self, cfg_path_region_refiner, models=None, boxes=None, is_rpn=False):
        self.cfg = yaml.load(open(cfg_path_region_refiner), Loader=yaml.FullLoader)
        if is_rpn:
            self.cfg = self.cfg['RPN']
        self.models = models
        self.boxes = boxes
        self.feat = None
        try:
            self.lambd = self.cfg['REGION_REFINER']['opts']['lambda']
        except:
            self.lambd = None
        self.sigma = None
        self.COXY = None
        self.stats = None
        self.is_rpn = is_rpn

    def loadRegionRefiner(self):
        return

    def trainRegionRefiner(self):
        trainer = RegionRefinerTrainer(self.cfg)
        trainer.lambd = self.lambd
        trainer.sigma = self.sigma
        trainer.COXY = self.COXY
        trainer.is_rpn = self.is_rpn
        self.models = trainer()
        return self.models

    def testRegionRefiner(self):
        return

    def predict(self):
        predictor = RegionPredictor(self.cfg, self.models, self.boxes)
        predictor.feat = self.feat
        if self.stats is not None:
            predictor.stats = self.stats
        refined_regions = predictor()
        return refined_regions
