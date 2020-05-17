import sys
import os

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
# sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, '..', '..')))

import RegionClassifierAbstract as rcA
# from utils import computeFeatStatistics, zScores
from scipy import stats
from scipy.io import loadmat


class OnlineRegionClassifier(rcA.RegionClassifierAbstract):

    def loadRegionClassifier(self) -> None:
        pass

    def selectPositives(self, dataset):
        positives_file = self.experiment_name + '_positives.mat'
        try:
            positives = loadmat(positives_file)
        except:
            print('To implement selectPositives in OnlineRegionClassifier')
            positives = None

        return positives

    def trainWithMinibootstrap(self, negatives, positives, opts):
        print('To implement trainWithMinibootstrap in OnlineRegionClassifier')

        # Concatenate postives and negatives samples
        dataset = dict()
        dataset['x_train'] = [positives.feat, negatives.feat]
        dataset['y_train'] = [positives.label, negatives.label]

        model = self.classifier.train(dataset, opts)
        return model

    def trainRegionClassifier(self, dataset, opts):

        negatives = self.negative_selector.selectNegatives(dataset, self.experiment_name, opts)
        positives = self.selectPositives(dataset)

        # [mean, mean_norm] = computeFeatStatistics(positives, negatives)
        # positives.feat = zScores(positives.feat, mean, mean_norm)
        # negatives.feat = zScores(negatives.feat, mean, mean_norm)
        positives.feat = stats.zscore(positives.feat)
        negatives.feat = stats.zscore(negatives.feat)

        model = self.trainWithMinibootstrap(negatives, positives, opts)

        return model

    def crossValRegionClassifier(self, dataset):
        pass

    def testRegionClassifier(self, dataset) -> None:
        pass

    def predict(self, dataset) -> None:
        pass
