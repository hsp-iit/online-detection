import sys
import os

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(basedir, '..', '..')))

import RegionClassifierAbstract as rcA
from utils import computeFeatStatistics, zScores, loadFeature
import h5py
import numpy as np
import torch
from maskrcnn_benchmark.structures.bounding_box import BoxList


class OnlineRegionClassifier(rcA.RegionClassifierAbstract):

    def loadRegionClassifier(self) -> None:
        pass

    def selectPositives(self, imset_path, opts):
        feat_path = os.path.join(basedir, '..', '..', '..', 'Data', 'feat_cache', self.experiment_name)
        positives_file = os.path.join(feat_path, self.experiment_name + '_positives.mat')
        try:
            mat_positives = h5py.File(positives_file, 'r')
            X_pos = mat_positives['X_pos']
            positives = []
            for i in range(opts['num_classes']-1):
                positives.append(mat_positives[X_pos[0, i]][()].transpose())
        except:
            with open(imset_path, 'r') as f:
                path_list = f.readlines()
            feat_path = os.path.join(basedir, '..', '..', '..', 'Data', 'feat_cache', self.experiment_name)
            for i in range(len(path_list)):
                l = loadFeature(feat_path, path_list[i])
                for c in opts['num_classes']:
                    if len(positives) < c + 1:
                        positives.append(0)  # Initialization for class c-th
                    sel = np.where(l['class'] == c + 1)[0]  # TO CHECK BECAUSE OF MATLAB 1
                                                            # INDEXING Moreover class 0 is bkg
                    if len(sel):
                        positives[c] = np.vstack(positives[c], l['feat'][sel, :])

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
        for i in range(opts['num_classes']-1):
            print('---------------------- Training Class number {} ----------------------'.format(i))
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
                    hard_idx = np.argwhere(neg_pred.numpy() > self.negative_selector.neg_hard_thresh)[:,0]
                    caches[i]['neg'] = np.vstack((caches[i]['neg'], negatives[i][j][hard_idx]))
                    print('Chosen {} hard negatives from the {}th batch'.format(len(hard_idx), j))

                print('Traning with {} positives and {} negatives'.format(len(caches[i]['pos']), len(caches[i]['neg'])))
                model[i] = self.updateModel(caches[i], opts)
                neg_pred = self.classifier.predict(model[i], caches[i]['neg'])  # To check

                easy_idx = np.argwhere(neg_pred.numpy() < self.negative_selector.neg_easy_thresh)[:,0]
                caches[i]['neg'] = np.delete(caches[i]['neg'], easy_idx, axis=0)
                print('Removed {} easy negatives. {} Remaining'.format(len(easy_idx), len(caches[i]['neg'])))

        model_name = 'model_' + self.experiment_name
        torch.save(model, model_name)
        return model

    def trainRegionClassifier(self, dataset, opts):

        negatives = self.negative_selector.selectNegatives(dataset, self.experiment_name, opts)
        positives = self.selectPositives(dataset, opts)

        self.mean, self.std, self.mean_norm = computeFeatStatistics(positives, negatives)
        for i in range(opts['num_classes']-1):
            positives[i] = zScores(positives[i], self.mean, self.mean_norm)
            for j in range(len(negatives[i])):
               negatives[i][j] = zScores(negatives[i][j], self.mean, self.mean_norm)

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
        # (https://github.com/fedeceola/online-object-segmentation/blob/33ab5c3d4986d1a0785c8ad7cfae8657f73670dd/maskrcnn_pytorch/benchmark/modeling/roi_heads/box_head/inference.py)
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
        # thresh = 0
        predictions = []
        # scores = np.zeros((len(boxes), opts['num_classes']))
        for i in range(len(path_list)):
            l = loadFeature(feat_path, path_list[i].rstrip())
            if l is not None:
                print('Processing image {}'.format(path_list[i]))
                I = np.nonzero(l['gt'] == 0)
                boxes = l['boxes'][I, :][0]
                X_test = l['feat'][I, :][0]
                X_test = zScores(X_test, self.mean, self.mean_norm)
                scores = - np.ones((len(boxes), opts['num_classes']))
                for c in range(0, opts['num_classes']-1):
                    pred = self.classifier.predict(model[c], X_test)
                    scores[:, c+1] = np.squeeze(pred.numpy())

                b = BoxList(torch.from_numpy(boxes), (640, 480), mode="xyxy")    # TO parametrize image shape
                b.add_field("scores", torch.from_numpy(np.float32(scores)))
                predictions.append(b)
            else:
                print('None feature loaded. Skipping image {}.'.format(path_list[i]))

        return scores, boxes, predictions

    def predict(self, dataset) -> None:
        pass
