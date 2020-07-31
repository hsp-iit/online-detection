from RegionRefinerAbstract import RegionRefinerAbstract

from region_refiner_trainer import RegionRefinerTrainer
from region_predictor import RegionPredictor
import yaml


class RegionRefiner(RegionRefinerAbstract):
    def __init__(self, cfg_path_region_refiner, models=None, boxes=None):
        self.cfg = yaml.load(open(cfg_path_region_refiner), Loader=yaml.FullLoader)
        self.models = models
        self.boxes = boxes
        try:
            self.lambd = self.cfg['REGION_REFINER']['opts']['lambda']
        except:
            self.lambd = None
        self.sigma = None
        self.COXY = None
        self.stats = None

    def loadRegionRefiner(self):
        return

    def trainRegionRefiner(self):
        trainer = RegionRefinerTrainer(self.cfg)
        trainer.lambd = self.lambd
        trainer.sigma = self.sigma
        trainer.COXY = self.COXY
        self.models = trainer()
        return self.models

    def testRegionRefiner(self):
        return

    def predict(self):
        predictor = RegionPredictor(self.cfg, self.models, self.boxes)
        if self.stats is not None:
            predictor.stats = self.stats
        refined_regions = predictor()
        return refined_regions
