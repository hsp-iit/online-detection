import sys
import os

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..')))

import RegionClassifierAbstract as rcA
from utils import computeFeatStatistics, zScores
from scipy import stats
import h5py
import numpy as np
import torch


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

    def updateModel(self, cache, opts):

        X_neg = cache['neg']
        X_pos = cache['pos']
        num_neg = len(X_neg)
        num_pos = len(X_pos)
        X = np.vstack((X_pos, X_neg))
        y = np.vstack((np.transpose(np.ones(num_pos)[np.newaxis]), -np.transpose(np.ones(num_neg)[np.newaxis])))

        return self.classifier.train(X, y, opts)

    def trainWithMinibootstrap(self, negatives, positives, opts):
        iterations = self.negative_selector.iterations
        caches = []
        model = []
        for i in range(opts['num_classes']):
            first_time = True
            for j in range(iterations):
                if first_time:
                    dataset = {}
                    dataset['pos'] = positives[i]
                    dataset['neg'] = negatives[i][j]
                    caches.append(dataset)
                    model.append(None)
                    first_time = False
                else:
                    neg_pred = self.classifier.predict(model[i], negatives[i][j])  # To check
                    hard_idx = np.argwhere(neg_pred < self.negative_selector.neg_hard_thresh)
                    caches[i]['neg'] = np.vstack((caches[i]['neg'], negatives[i][j][hard_idx.numpy()[0]]))

                model[i] = self.updateModel(caches[i], opts)
                neg_pred = self.classifier.predict(model[i], caches[i]['neg'])  # To check

                easy_idx = np.argwhere(neg_pred > self.negative_selector.neg_easy_thresh)
                caches[i]['neg'] = np.delete(caches[i]['neg'], easy_idx, axis=0)
        model_name = 'model_' + self.experiment_name
        torch.save(model, model_name)
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

    def loadFeature(self, feat_path, img, type='mat'):
        feat_file = None
        if type == 'mat':
            file_name = img + '.mat'
            feat_file = h5py.File(os.path.join(feat_path, file_name), 'r')
        else:
            print('Unrecognized type file: {}'.format(type))

        return feat_file

    def testRegionClassifier(self, model, imset_path, opts):
        # What do we need?
        # X list of images of the test set
        # X where to find features
        # - where to find annotations
        # X Falkon model to test
        # - max_per_set and max_per_image (?)
        # - La threshold per le predizioni dove avviene?
        # - Nel matlab ogni classificatore dava la propria predizione per ogni box e
        #   tutte queste predizioni venivano fornite al devkit nella forma aboxes{i}{j} i-esima immagine j-esima classe.
        #   In questo caso come funziona?

        # Options setting
        with open(imset_path, 'r') as f:
            path_list = f.readlines()
        feat_path = os.path.join(basedir, '..', '..', '..', 'Data', 'feat_cache', self.experiment_name)

        # Loop on the dataset:
        # - Retrieve feature
        # - Normalize feature
        # - For each class
        # -- Test of the model
        # -- Storage of the predictions in mask r-cnn format
        # -- Processing of the predictions (?)
        thresh = 0
        predictions = {}
        for i in range(len(path_list)):
            l = self.loadFeature(feat_path, path_list[i])
            if l is not None:
                X_test = l['feat']
                for c in range(opts['num_classes']):
                    boxes = l['boxes']
                    pred = self.classifier.predict(model[i], X_test)
                    I = np.nonzero(l['gt'] == 0 & pred > thresh)
                    boxes = boxes(I)
                    scores = pred(I)
                    # Concatenate obtained predictions with the old ones

            else:
                print('None feature loaded. Skipping image {}.'.format(path_list[i]))

        # Evaluation of the prediction using Mask R-CNN evaluation function
        result = 0

        return result

    def predict(self, dataset) -> None:
        pass
