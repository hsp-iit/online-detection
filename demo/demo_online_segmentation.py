import sys
import os
basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'region-classifier')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'region-refiner')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'feature-extractor')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'accuracy-evaluator')))


from mrcnn_modified.config import cfg
from predictor_online_segmentation import OnlineSegmentationDemo
import cv2
import random

config_file = os.path.abspath(os.path.join(basedir, os.path.pardir, "experiments", "configs", "config_segmentation_elsa.yaml"))
run_single_image = False
# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
#cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
#cfg.MODEL.WEIGHT = "/home/IIT.LOCAL/fceola/workspace/ws_mask/maskrcnn-benchmark/experiments/e2e_mask_rcnn_mask_on_imagenet_R_50_FPN_1x_IVOS_segmentation_fine_tune_no_FPN/model_final.pth"

coco_demo = OnlineSegmentationDemo(
    cfg,
    confidence_threshold=0.3,
    show_mask_heatmaps=False
)
# load image and then run prediction
if run_single_image:
    image_path = '/home/iiticublap205/IIT/datasets/iCWT/TABLE-TOP-single-object-masks/test/Images/mug3/00000144.jpg'
    image = cv2.imread(image_path,1)
    predictions = coco_demo.run_on_opencv_image(image)
    cv2.imshow('predictions', predictions)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    quit()

else: #run on all test dataset
    #with open("/home/iiticublap205/IIT/datasets/iCWT/TABLE-TOP-single-object-masks/test/ImageSets/test_AutomSegm_tabletop_21objs.txt") as f:
    with open(os.path.abspath(os.path.join(basedir, os.path.pardir, "Data", "datasets", "iCWT", "TABLE-TOP-single-object-masks", "test", "ImageSets", "test_AutomSegm_tabletop_21objs.txt"))) as f:
        ids = f.readlines()
    ids = [x.strip("\n") for x in ids]    
    #ids_to_remove = [5004, 5005, 5018, 5055, 5079, 5081, 5087, 5088, 5092, 5123, 5143, 5149, 5157, 5166, 5184, 5189, 5828, 5831, 5929, 5964, 6124, 7869, 7917, 7948, 7969, 9847, 10413, 10420, 10482, 10571, 10800]
    #for i in reversed(ids_to_remove):
    #    del ids[i]
    random.seed(1)
    ids = random.sample(ids, round(len(ids)*10/100))
    for i in range(len(ids)):
        #image = cv2.imread('/home/iiticublap205/IIT/datasets/iCWT/TABLE-TOP-single-object-masks/test/Images/{}.jpg'.format(ids[i]),1)
        image = cv2.imread(os.path.abspath(os.path.join(basedir, os.path.pardir, "Data", "datasets", "iCWT", "TABLE-TOP-single-object-masks", "test", "Images", "{}.jpg".format(ids[i]))),1)
        try:
            predictions = coco_demo.run_on_opencv_image(image)
        except:
            continue
        if not os.path.exists('test_masks_oos'):
            os.makedirs('test_masks_oos')
        cv2.imwrite('test_masks_oos/{}.jpg'.format(i), predictions)

