from RegionRefinerAbstract import RegionRefinerAbstract

from region_refiner_trainer import RegionRefinerTrainer
from region_predictor import RegionPredictor
import yaml


class RegionRefiner(RegionRefinerAbstract):
    def __init__(self, cfg_path_region_refiner, is_rpn=False):
        self.cfg = yaml.load(open(cfg_path_region_refiner), Loader=yaml.FullLoader)
        if is_rpn:
            self.cfg = self.cfg['RPN']
        try:
            self.lambd = self.cfg['REGION_REFINER']['opts']['lambda']
        except:
            self.lambd = None
        self.is_rpn = is_rpn

    def loadRegionRefiner(self):
        return

    def trainRegionRefiner(self, COXY, output_dir=None):
        trainer = RegionRefinerTrainer(self.cfg, lmbd=self.cfg['REGION_REFINER']['opts']['lambda'], is_rpn=self.is_rpn)
        self.models = trainer(COXY, output_dir=output_dir)
        return self.models

    def testRegionRefiner(self):
        return

    def predict(self, boxes, features, models=None, normalize_features=False, stats=None):
        if models is None:
            predictor = RegionPredictor(self.cfg, self.models)
        else:
            predictor = RegionPredictor(self.cfg, models)
        refined_regions = predictor(boxes, features, normalize_features=normalize_features, stats=stats)
        return refined_regions
