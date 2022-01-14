import sys
import os

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, os.path.pardir)))

import RegionClassifierAbstract as rcA
from py_od_utils import computeFeatStatistics
import numpy as np
import torch
from maskrcnn_benchmark.structures.bounding_box import BoxList
import time

import yaml
import copy

from joblib import Parallel, delayed, parallel_backend
import multiprocessing


class OnlineRegionClassifier(rcA.RegionClassifierAbstract):

    def __init__(self, classifier, positives, negatives, stats=None, cfg_path=None, is_rpn=False, is_segmentation=False):
        if cfg_path is not None:
            self.cfg = yaml.load(open(cfg_path), Loader=yaml.FullLoader)
            if is_rpn:
                self.cfg = self.cfg['RPN']
            if not is_segmentation:
                self.classifier_options = self.cfg['ONLINE_REGION_CLASSIFIER']['CLASSIFIER']
                self.lam = self.cfg['ONLINE_REGION_CLASSIFIER']['CLASSIFIER']['lambda']
                self.sigma = self.cfg['ONLINE_REGION_CLASSIFIER']['CLASSIFIER']['sigma']
                self.hard_tresh = self.cfg['ONLINE_REGION_CLASSIFIER']['MINIBOOTSTRAP']['HARD_THRESH']
                self.easy_tresh = self.cfg['ONLINE_REGION_CLASSIFIER']['MINIBOOTSTRAP']['EASY_THRESH']
            else:
                self.classifier_options = self.cfg['ONLINE_SEGMENTATION']['CLASSIFIER']
                self.lam = self.cfg['ONLINE_SEGMENTATION']['CLASSIFIER']['lambda']
                self.sigma = self.cfg['ONLINE_SEGMENTATION']['CLASSIFIER']['sigma']
                self.hard_tresh = self.cfg['ONLINE_SEGMENTATION']['MINIBOOTSTRAP']['HARD_THRESH']
                self.easy_tresh = self.cfg['ONLINE_SEGMENTATION']['MINIBOOTSTRAP']['EASY_THRESH']
            self.mean = 0
            self.std = 0
            self.mean_norm = 0
            self.is_rpn = is_rpn


        else:
            print('Config file path not given. cfg variable set to None.')
            self.cfg = None

        self.classifier = classifier
        self.negatives = negatives
        self.positives = positives
        self.num_classes = len(self.cfg['CHOSEN_CLASSES'])
        if is_rpn:
            self.num_classes += 1
        if stats:
            self.stats = stats
            self.mean = self.stats['mean']
            self.std = self.stats['std']
            self.mean_norm = self.stats['mean_norm']

        self.normalized = False
        self.is_segmentation = is_segmentation

        self.return_caches = False


    def loadRegionClassifier(self) -> None:
        pass

    def processOptions(self, opts):
        if 'num_classes' in opts:
            self.num_classes = opts['num_classes']
        if 'imset_train' in opts:
            self.train_imset = opts['imset_train']
        if 'classifier_options' in opts:
            self.classifier_options = opts['classifier_options']
        if 'is_rpn' in opts:
            self.is_rpn = opts['is_rpn']
        if 'lam' in opts:
            self.lam = opts['lam']
        if 'sigma' in opts:
            self.sigma = opts['sigma']
        if 'return_caches' in opts:
            self.return_caches = opts['return_caches']
        if 'normalized' in opts:
            self.normalized = opts['normalized']


    def updateModel(self, cache):
        X_neg = cache['neg']
        X_pos = cache['pos']
        num_neg = len(X_neg)
        num_pos = len(X_pos)
        X = torch.cat((X_pos, X_neg), 0)
        y = torch.cat((torch.transpose(torch.ones(num_pos, device='cuda'), 0, 0), -torch.transpose(torch.ones(num_neg, device='cuda'), 0, 0)), 0)

        if self.sigma is not None and self.lam is not None:
            print('Updating model with lambda: {} and sigma: {}'.format(self.lam, self.sigma))
            return self.classifier.train(X, y, sigma=self.sigma, lam=self.lam)
        else:
            print('Updating model with default lambda and sigma')
            return self.classifier.train(X, y)


    def trainWithMinibootstrap(self, negatives, positives, output_dir=None):
        caches = []
        model = []
        t = time.time()
        for i in range(self.num_classes-1):
            if (len(positives[i]) != 0) & (len(negatives[i]) != 0):
                print('---------------------- Training Class number {} ----------------------'.format(i))
                first_time = True
                for j in range(len(negatives[i])):
                    t_iter = time.time()
                    if first_time:
                        dataset = {}
                        dataset['pos'] = positives[i]
                        dataset['neg'] = negatives[i][j]
                        caches.append(dataset)
                        model.append(None)
                        first_time = False
                    else:
                        t_hard = time.time()
                        neg_pred = self.classifier.predict(model[i], negatives[i][j])
                        hard_idx = torch.where(neg_pred > self.hard_tresh)[0]
                        caches[i]['neg'] = torch.cat((caches[i]['neg'], negatives[i][j][hard_idx]), 0)
                        print('Hard negatives selected in {} seconds'.format(time.time() - t_hard))
                        print('Chosen {} hard negatives from the {}th batch'.format(len(hard_idx), j))

                    print('Traning with {} positives and {} negatives'.format(len(caches[i]['pos']), len(caches[i]['neg'])))
                    t_update = time.time()
                    model[i] = self.updateModel(caches[i])
                    print('Model updated in {} seconds'.format(time.time() - t_update))

                    t_easy = time.time()
                    if len(caches[i]['neg']) != 0 and not j == len(negatives[i]) - 1:
                        neg_pred = self.classifier.predict(model[i], caches[i]['neg'])
                        keep_idx = torch.where(neg_pred >= self.easy_tresh)[0]
                        easy_idx = len(caches[i]['neg']) - len(keep_idx)
                        caches[i]['neg'] = caches[i]['neg'][keep_idx]
                        print('Easy negatives selected in {} seconds'.format(time.time() - t_easy))
                        print('Removed {} easy negatives. {} Remaining'.format(easy_idx, len(caches[i]['neg'])))
                        print('Iteration {}th done in {} seconds'.format(j, time.time() - t_iter))
                    # Delete cache of the i-th classifier if it is the last iteration to free memory
                    if j == len(negatives[i]) - 1 and not self.return_caches:
                        caches[i] = None
                        torch.cuda.empty_cache()
            else:
                model.append(None)
                dataset = {}
                caches.append(dataset)

        training_time = time.time() - t
        print('Online Classifier trained in {} seconds'.format(training_time))
        if output_dir and self.is_rpn:
            with open(os.path.join(output_dir, "result.txt"), "a") as fid:
                fid.write("RPN's Online Classifier training time: {}min:{}s \n".format(int(training_time/60), round(training_time%60)))
        elif output_dir and self.is_segmentation:
            with open(os.path.join(output_dir, "result.txt"), "a") as fid:
                fid.write("Online Segmentation training time: {}min:{}s \n".format(int(training_time/60), round(training_time%60)))
        elif output_dir and not self.is_rpn and not self.is_segmentation:
            with open(os.path.join(output_dir, "result.txt"), "a") as fid:
                fid.write("Detector's Online Classifier training time: {}min:{}s \n".format(int(training_time/60), round(training_time%60)))
        if self.return_caches:
            self.caches = caches
        return model

    def updateModel_parallel(self, cache, classifier):
        X_neg = cache['neg']
        X_pos = cache['pos']
        num_neg = len(X_neg)
        num_pos = len(X_pos)
        X = torch.cat((X_pos, X_neg), 0)
        y = torch.cat((torch.transpose(torch.ones(num_pos, device='cuda'), 0, 0), -torch.transpose(torch.ones(num_neg, device='cuda'), 0, 0)), 0)

        if self.sigma is not None and self.lam is not None:
            print('Updating model with lambda: {} and sigma: {}'.format(self.lam, self.sigma))
            return classifier.train(X, y, sigma=self.sigma, lam=self.lam)
        else:
            print('Updating model with default lambda and sigma')
            return classifier.train(X, y)

    def compute_model_parallel(self, negatives_i, positives_i, i, classifier):
        caches_i ={}
        model_i = None

        if (len(positives_i) != 0) & (len(negatives_i) != 0):
            print('---------------------- Training Class number {} ----------------------'.format(i))
            first_time = True
            for j in range(len(negatives_i)):
                print(i, j, "!!!")
                t_iter = time.time()
                if first_time:
                    caches_i['pos'] = positives_i
                    caches_i['neg'] = negatives_i[j]
                    print(first_time)
                    first_time = False
                else:
                    t_hard = time.time()
                    neg_pred = classifier.predict(model_i, negatives_i[j])
                    hard_idx = torch.where(neg_pred > self.hard_tresh)[0]
                    caches_i['neg'] = torch.cat((caches_i['neg'], negatives_i[j][hard_idx]), 0)
                    print('Hard negatives selected in {} seconds'.format(time.time() - t_hard))
                    print('Chosen {} hard negatives from the {}th batch'.format(len(hard_idx), j))

                print('Traning with {} positives and {} negatives'.format(len(caches_i['pos']), len(caches_i['neg'])))
                t_update = time.time()
                model_i = self.updateModel_parallel(caches_i, classifier)
                print('Model updated in {} seconds'.format(time.time() - t_update))

                t_easy = time.time()
                if len(caches_i['neg']) != 0 and not j == len(negatives_i) - 1:
                    neg_pred = classifier.predict(model_i, caches_i['neg'])
                    keep_idx = torch.where(neg_pred >= self.easy_tresh)[0]
                    easy_idx = len(caches_i['neg']) - len(keep_idx)
                    caches_i['neg'] = caches_i['neg'][keep_idx]
                    print('Easy negatives selected in {} seconds'.format(time.time() - t_easy))
                    print('Removed {} easy negatives. {} Remaining'.format(easy_idx, len(caches_i['neg'])))
                    print('Iteration {}th done in {} seconds'.format(j, time.time() - t_iter))
                # Delete cache of the i-th classifier if it is the last iteration to free memory
                if j == len(negatives_i) - 1 and not self.return_caches:
                    caches_i = None
                    torch.cuda.empty_cache()

        return model_i, caches_i

    def trainWithMinibootstrapParallel(self, negatives, positives, output_dir=None):
        caches = []
        models = []
        t = time.time()
        classifier = copy.deepcopy(self.classifier)
        #models, caches = Parallel(n_jobs=1, prefer="threads", verbose=1)(delayed(self.compute_model_parallel)(negatives[i], positives[i], i, copy.deepcopy(classifier)) for i in range(self.num_classes-1))
        training_output = Parallel(n_jobs=2, prefer="threads", verbose=1)(delayed(self.compute_model_parallel)(negatives[i], positives[i], i, copy.deepcopy(classifier)) for i in range(self.num_classes-1))
        for i in range(len(training_output)):
            models.append(training_output[i][0])
            caches.append(training_output[i][1])

        training_time = time.time() - t
        print('Online Classifier trained in {} seconds'.format(training_time))
        if output_dir and self.is_rpn:
            with open(os.path.join(output_dir, "result.txt"), "a") as fid:
                fid.write("RPN's Online Classifier training time: {}min:{}s \n".format(int(training_time/60), round(training_time%60)))
        elif output_dir and self.is_segmentation:
            with open(os.path.join(output_dir, "result.txt"), "a") as fid:
                fid.write("Online Segmentation training time: {}min:{}s \n".format(int(training_time/60), round(training_time%60)))
        elif output_dir and not self.is_rpn and not self.is_segmentation:
            with open(os.path.join(output_dir, "result.txt"), "a") as fid:
                fid.write("Detector's Online Classifier training time: {}min:{}s \n".format(int(training_time/60), round(training_time%60)))
        if self.return_caches:
            self.caches = caches
        return models


    def trainRegionClassifier(self, opts=None, output_dir=None):
        if opts is not None:
            self.processOptions(opts)
        print('Training Online Region Classifier')
        negatives = self.negatives
        positives = self.positives

        if not self.normalized:
            for i in range(self.num_classes-1):
                if len(positives[i]):
                    positives[i] = self.zScores(positives[i])
                for j in range(len(negatives[i])):
                    if len(negatives[i][j]):
                        negatives[i][j] = self.zScores(negatives[i][j])
            self.normalized = True

        model = self.trainWithMinibootstrap(negatives, positives, output_dir=output_dir)

        if not self.return_caches:
            return model
        else:
            return model, self.caches

    def testRegionClassifier(self, model, test_boxes):
        print('Online Region Classifier testing')
        predictions = []
        total_testing_time = 0
        try:
            for c in range(0, self.num_classes-1):
                model[c].ny_points_ = model[c].ny_points_.to('cuda')
                model[c].alpha_ = model[c].alpha_.to('cuda')
        except:
            pass
        for i in range(len(test_boxes)):
            l = test_boxes[i]
            if l is not None:
                I = np.nonzero(l['gt'] == 0)
                boxes = l['boxes'][I, :][0]
                X_test = torch.tensor(l['feat'][I, :][0], device='cuda')
                t0 = time.time()
                if self.mean_norm != 0:
                   X_test = self.zScores(X_test)
                scores = - torch.ones((len(boxes), self.num_classes))
                for c in range(0, self.num_classes-1):
                    pred = self.classifier.predict(model[c], X_test)
                    scores[:, c+1] = torch.squeeze(pred)

                total_testing_time = total_testing_time + time.time() - t0
                b = BoxList(torch.from_numpy(boxes), (l['img_size'][0], l['img_size'][1]), mode="xyxy")
                b.add_field("scores", scores.to('cpu'))
                predictions.append(b)

        avg_time = total_testing_time/len(test_boxes)
        print('Average image testing time: {} seconds.'.format(avg_time))
        return predictions
    
    def predict(self, dataset) -> None:
        pass

    def zScores(self, feat, target_norm=20):
        feat = feat - self.mean
        feat = feat * (target_norm / self.mean_norm.item())
        return feat
