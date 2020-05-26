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
import ntpath
import time


class OnlineRegionClassifier(rcA.RegionClassifierAbstract):

    def loadRegionClassifier(self) -> None:
        pass

    def processOptions(self, opts):
        if 'num_classes' in opts:
            self.num_classes = opts['num_classes']
        if 'imset_train' in opts:
            self.train_imset = opts['imset_train']
        if 'classifier_options' in opts:
            self.classifier_options = opts['classifier_options']

    def selectPositives(self, feat_type='h5'):
        feat_path = os.path.join(basedir, '..', '..', '..', 'Data', 'feat_cache', self.feature_folder)
        positives_file = os.path.join(feat_path, 'positives')
        try:
            if feat_type == 'mat':
                mat_positives = h5py.File(positives_file, 'r')
                X_pos = mat_positives['X_pos']
                positives = []
                for i in range(self.num_classes-1):
                    positives.append(mat_positives[X_pos[0, i]][()].transpose())
            elif feat_type == 'h5':
                positives_dataset = h5py.File(positives_file, 'r')['positives']
                positives = []
                for i in range(self.num_classes-1):
                    positives.append(positives_dataset[i])
            else:
                print('Unrecognized type of feature file')
                positives = None
        except:
            with open(self.train_imset, 'r') as f:
                path_list = f.readlines()
            feat_path = os.path.join(basedir, '..', '..', '..', 'Data', 'feat_cache', self.feature_folder)
            positives = []
            for i in range(len(path_list)):
                l = loadFeature(feat_path, path_list[i].rstrip())
                for c in range(self.num_classes - 1):
                    if len(positives) < c + 1:
                        positives.append([])  # Initialization for class c-th
                    sel = np.where(l['class'] == c + 1)[0]  # TO CHECK BECAUSE OF MATLAB 1
                                                            # INDEXING Moreover class 0 is bkg
                    if len(sel):
                        if len(positives[c]) == 0:
                            positives[c] = l['feat'][sel, :]
                        else:
                            positives[c] = np.vstack((positives[c], l['feat'][sel, :]))
            hf = h5py.File(positives_file, 'w')
            hf.create_dataset('positives', data=positives)
            hf.close()

        return positives

    def updateModel(self, cache):
        X_neg = cache['neg']
        X_pos = cache['pos']
        num_neg = len(X_neg)
        num_pos = len(X_pos)
        X = np.vstack((X_pos, X_neg))
        y = np.vstack((np.transpose(np.ones(num_pos)[np.newaxis]), -np.transpose(np.ones(num_neg)[np.newaxis])))

        return self.classifier.train(X, y, self.classifier_options)

    def trainWithMinibootstrap(self, negatives, positives):
        iterations = self.negative_selector.iterations
        caches = []
        model = []
        t = time.time()
        for i in range(self.num_classes-1):
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
                model[i] = self.updateModel(caches[i])
                neg_pred = self.classifier.predict(model[i], caches[i]['neg'])  # To check

                easy_idx = np.argwhere(neg_pred.numpy() < self.negative_selector.neg_easy_thresh)[:,0]
                caches[i]['neg'] = np.delete(caches[i]['neg'], easy_idx, axis=0)
                print('Removed {} easy negatives. {} Remaining'.format(len(easy_idx), len(caches[i]['neg'])))

        training_time = time.time() - t
        print('Online Classifier trained in {} seconds'.format(training_time))
        model_name = 'model_' + self.experiment_name
        torch.save(model, model_name)
        return model

    def trainRegionClassifier(self, opts=None):
        if opts is not None:
            self.processOptions(opts)
        print('Training Online Region Classifier')
        # Still to implement early stopping of negatives selection
        negatives = self.negative_selector.selectNegatives()
        positives = self.selectPositives()

        self.mean, self.std, self.mean_norm = computeFeatStatistics(positives, negatives, self.feature_folder)
        for i in range(self.num_classes-1):
            positives[i] = zScores(positives[i], self.mean, self.mean_norm)
            for j in range(len(negatives[i])):
               negatives[i][j] = zScores(negatives[i][j], self.mean, self.mean_norm)

        model = self.trainWithMinibootstrap(negatives, positives)

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

    def testRegionClassifier(self, model):
        print('Online Region Classifier testing')
        with open(self.test_imset, 'r') as f:
            path_list = f.readlines()
        feat_path = os.path.join(basedir, '..', '..', '..', 'Data', 'feat_cache', self.feature_folder)

        predictions = []
        total_testing_time = 0
        for i in range(len(path_list)):
            print('Testing {}/{} : {}'.format(i, len(path_list), path_list[i].rstrip()))
            l = loadFeature(feat_path, path_list[i].rstrip())
            if l is not None:
                print('Processing image {}'.format(path_list[i]))
                I = np.nonzero(l['gt'] == 0)
                boxes = l['boxes'][I, :][0]
                X_test = l['feat'][I, :][0]
                t0 = time.time()
                if self.mean_norm != 0:
                   X_test = zScores(X_test, self.mean, self.mean_norm)
                scores = - np.ones((len(boxes), self.num_classes))
                for c in range(0, self.num_classes-1):
                    pred = self.classifier.predict(model[c], X_test)
                    scores[:, c+1] = np.squeeze(pred.numpy())

                total_testing_time = total_testing_time + t0 - time.time()
                b = BoxList(torch.from_numpy(boxes), (640, 480), mode="xyxy")    # TO parametrize image shape
                b.add_field("scores", torch.from_numpy(np.float32(scores)))
                b.add_field("name_file", path_list[i].rstrip())
                predictions.append(b)
            else:
                print('None feature loaded. Skipping image {}.'.format(path_list[i]))

        avg_time = total_testing_time/len(path_list)
        print('Testing an image in {} seconds.'.format(avg_time))
        return predictions

    def predict(self, dataset) -> None:
        pass
