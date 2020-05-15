from trainer_feature_task import TrainerFeatureTask
from feature_extractor_detector import FeatureExtractorDetector


class FeatureExtractor:
    def __init__(self, cfg_path_feature_task=None, cfg_path_target_task=None):
        self.cfg_path_feature_task = cfg_path_feature_task
        self.cfg_path_target_task = cfg_path_target_task

    def train_model_on_feature_task(self):
        # call class to train from scratch a model on the feature task
        trainer = TrainerFeatureTask(self.cfg_path_feature_task)
        models = trainer()

        # maybe this part can be included in the training class
        if len(models) == 0:
            model = models[0]
        else:
            model = self.select_feature_extractor(models)

        return model   #or return path to models if they are saved somewhere and maybe some metrics such as mAP

    def select_feature_extractor(self, models):
        # evaluate models computed by train_model_on_feature_task, maybe using some metrics learning
        model = []

        return model

    def extract_rpn_features(self):
        # call class to extract rpn features:
        features = []

        return features

    def extract_detector_features_from_feature_task_RPN(self):
        # call class to extract detector features:
        features = []

        return features

    def extract_detector_features_from_updated_RPN(self):
        # call class to extract detector features embedding rpn update:
        feature_extractor = FeatureExtractorDetector(self.cfg_path_feature_task)#TODO change path to target task
        features = feature_extractor()

        return features

a = FeatureExtractor("../configs/e2e_mask_rcnn_mask_off_imagenet_R_50_FPN_1x_online_object_detection_feature_task_no_FPN.yaml")
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#models = a.extract_detector_features_from_updated_RPN()
=======
models = a.extract_detector_features_from_updated_RPN()
>>>>>>> Initial feature-extractor scripts
=======
#models = a.extract_detector_features_from_updated_RPN()
>>>>>>> Minor changes
=======
models = a.extract_detector_features_from_updated_RPN()
>>>>>>> Initial feature-extractor scripts

models = a.train_model_on_feature_task()
print(models)
