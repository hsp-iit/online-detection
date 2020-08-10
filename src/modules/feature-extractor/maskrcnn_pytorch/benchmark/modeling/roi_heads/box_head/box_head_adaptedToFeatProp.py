# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Adapted but then split to a new file box_head_getProposals.py so probably deprecated
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator

# to save objects
import pickle
# to save as .mat
# TODO ms-thesis-segmentation important!! install scipy to your virtual environment
import scipy.io


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)

        # TODO ms-thesis-segmentation Save x and proposals to .mat file and save features as it is using https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        # save x and proposals
        arr_x = x.numpy()
        # TODO ms-thesis-segmentation only save the proposals, ignore objectness?
        arr_proposals = proposals[0].bbox.numpy()
        scipy.io.savemat('prop_features.mat', {'Features': arr_x, 'Proposals': arr_proposals})

        # save features
        with open('features.pkl', 'wb') as f:  # Python 2: open(..., 'w')
            pickle.dump(features, f)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            # TODO ms-thesis-segmentation for testing without the SVM module save result to test whether segmentation module works
            with open('result.pkl', 'wb') as f:  # Python 2: open(..., 'w')
                pickle.dump(result, f)
            # TODO simulate the input the SVM module will give
            bbox_arr = result[0].bbox.numpy()
            labels_arr = result[0].extra_fields['labels'].numpy()
            scores_arr = result[0].extra_fields['scores'].numpy()
            scipy.io.savemat('simulated_SVM_output.mat', {'BBox': bbox_arr, 'labels': labels_arr, 'scores': scores_arr})

            # TODO ms-thesis-segmentation result will need to be replaced by the output of the SVM module in the end
            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
