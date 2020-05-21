from src.modules.RegionRefinerAbstract import RegionRefinerAbstract

from region_refiner_trainer import RegionRefinerTrainer
import yaml

class RegionRefiner(RegionRefinerAbstract):
    def __init__(self, cfg_path_region_refiner = None):
        self.cfg = yaml.load(open(cfg_path_region_refiner), Loader=yaml.FullLoader)

    def loadRegionRefiner(self):
        return

    def trainRegionRefiner(self):
        trainer = RegionRefinerTrainer(self.cfg)
        trainer()
        return

    def testRegionRefiner(self):
        return

    def predict(self):
        return
