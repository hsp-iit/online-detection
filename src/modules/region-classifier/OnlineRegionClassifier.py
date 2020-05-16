import sys
import os

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, '..')))
import RegionClassifierAbstract as rcA


class OnlineRegionClassifier(rcA.RegionClassifierAbstract):
    def __init__(self):
        pass

    def loadRegionClassifier(self) -> None:
        pass

    def trainRegionClassifier(self, dataset) -> None:
        pass

    def testRegionClassifier(self, dataset) -> None:
        pass

    def predict(self, dataset) -> None:
        pass
