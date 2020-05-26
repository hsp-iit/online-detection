from src.modules.RegionRefinerAbstract import RegionRefinerAbstract

from region_refiner_trainer import RegionRefinerTrainer
from region_predictor import RegionPredictor
import yaml

class RegionRefiner(RegionRefinerAbstract):
    def __init__(self, cfg_path_region_refiner, models=None, boxes=None):
        self.cfg = yaml.load(open(cfg_path_region_refiner), Loader=yaml.FullLoader)
        self.models = models
        self.boxes = boxes

    def loadRegionRefiner(self):
        return

    def trainRegionRefiner(self):
        trainer = RegionRefinerTrainer(self.cfg)
        self.models = trainer()
        return

    def testRegionRefiner(self):
        return

    def predict(self):
        predictor = RegionPredictor(self.cfg, self.models, self.boxes)
        refined_regions = predictor()
        return refined_regions
