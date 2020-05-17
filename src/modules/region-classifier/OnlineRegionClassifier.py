import sys
import os

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, '..', '..')))

import RegionClassifierAbstract as rcA
from utils import computeFeatStatistics, zScores


class OnlineRegionClassifier(rcA.RegionClassifierAbstract):
    def __init__(self, classifier, negative_selector):
        self.classifier = classifier
        self.negative_selector = negative_selector

    def loadRegionClassifier(self) -> None:
        pass

    def selectPositives(self, dataset):
        print('To implement selectPositives in OnlineRegionClassifier')
        return dataset

    def trainWithMinibootstrap(self, negatives, postives, opts):
        print('To implement trainWithMinibootstrap in OnlineRegionClassifier')

        # Concatenate postives and negatives samples
        dataset = dict()
        dataset['x_train'] = [postives.feat, negatives.feat]
        dataset['y_train'] = [postives.label, negatives.label]

        model = self.classifier.train(dataset, opts)
        return model

    def trainRegionClassifier(self, dataset, opts):

        negatives = self.negative_selector.selectNegatives(dataset)
        positives = self.selectPositives(dataset)

        # TO IMPLEMENT
        [mean, mean_norm] = computeFeatStatistics(positives, negatives)
        positives.feat = zScores(positives.feat, mean, mean_norm)
        negatives.feat = zScores(negatives.feat, mean, mean_norm)

        model = self.trainWithMinibootstrap(negatives, positives, opts)

        return model

    def crossValRegionClassifier(self, dataset):
        pass

    def testRegionClassifier(self, dataset) -> None:
        pass

    def predict(self, dataset) -> None:
        pass
