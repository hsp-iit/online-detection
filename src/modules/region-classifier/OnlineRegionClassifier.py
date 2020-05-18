import sys
import os

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..')))

import RegionClassifierAbstract as rcA
from utils import computeFeatStatistics, zScores
from scipy import stats
import h5py


class OnlineRegionClassifier(rcA.RegionClassifierAbstract):

    def loadRegionClassifier(self) -> None:
        pass

    def selectPositives(self, dataset, opts):
        feat_path = os.path.join(basedir, '..', '..', '..', 'Data', 'feat_cache', self.experiment_name)
        positives_file = os.path.join(feat_path, self.experiment_name + '_positives.mat')
        try:
            mat_positives = h5py.File(positives_file, 'r')
            X_pos = mat_positives['X_pos']
            positives = []
            for i in range(opts['num_classes']):
                positives.append(mat_positives[X_pos[0, i]][()].transpose())
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
        positives = self.selectPositives(dataset, opts)

        mean, std, mean_norm = computeFeatStatistics(positives, negatives)
        for i in range(opts['num_classes']):
            positives[i] = zScores(positives[i], mean, mean_norm)
            negatives[i] = zScores(negatives[i], mean, mean_norm)

        model = self.trainWithMinibootstrap(negatives, positives, opts)

        return model

    def crossValRegionClassifier(self, dataset):
        pass

    def testRegionClassifier(self, dataset) -> None:
        pass

    def predict(self, dataset) -> None:
        pass
