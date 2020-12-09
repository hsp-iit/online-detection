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
import glob

#config_file = os.path.abspath(os.path.join(basedir, os.path.pardir, "experiments", "configs", "config_segmentation_elsa.yaml"))
config_file = os.path.abspath(os.path.join(basedir, os.path.pardir, "experiments", "configs", "config_segmentation_ycb_demo.yaml"))
run_single_image = True
# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
#cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
#cfg.MODEL.WEIGHT = "/home/IIT.LOCAL/fceola/workspace/ws_mask/maskrcnn-benchmark/experiments/e2e_mask_rcnn_mask_on_imagenet_R_50_FPN_1x_IVOS_segmentation_fine_tune_no_FPN/model_final.pth"

coco_demo = OnlineSegmentationDemo(
    cfg,
    confidence_threshold=0.2,
    show_mask_heatmaps=False
)

#images = sorted(glob.glob('/home/IIT.LOCAL/fceola/workspace/ws_papers_repos/YCB-Video/test/000048/rgb/*.png'))
"""
images = ["/home/IIT.LOCAL/fceola/workspace/ws_mask/corl-code/python-online-detection/Data/datasets/YCB-Video/test/000055/rgb/000588.png",
"/home/IIT.LOCAL/fceola/workspace/ws_mask/corl-code/python-online-detection/Data/datasets/YCB-Video/test/000051/rgb/001536.png",
"/home/IIT.LOCAL/fceola/workspace/ws_mask/corl-code/python-online-detection/Data/datasets/YCB-Video/test/000054/rgb/000445.png",
"/home/IIT.LOCAL/fceola/workspace/ws_mask/corl-code/python-online-detection/Data/datasets/YCB-Video/test/000058/rgb/000068.png",
"/home/IIT.LOCAL/fceola/workspace/ws_mask/corl-code/python-online-detection/Data/datasets/YCB-Video/test/000055/rgb/001358.png",
"/home/IIT.LOCAL/fceola/workspace/ws_mask/corl-code/python-online-detection/Data/datasets/YCB-Video/test/000049/rgb/000521.png",
"/home/IIT.LOCAL/fceola/workspace/ws_mask/corl-code/python-online-detection/Data/datasets/YCB-Video/test/000056/rgb/000143.png",
"/home/IIT.LOCAL/fceola/workspace/ws_mask/corl-code/python-online-detection/Data/datasets/YCB-Video/test/000048/rgb/000112.png",
"/home/IIT.LOCAL/fceola/workspace/ws_mask/corl-code/python-online-detection/Data/datasets/YCB-Video/test/000055/rgb/000415.png",
"/home/IIT.LOCAL/fceola/workspace/ws_mask/corl-code/python-online-detection/Data/datasets/YCB-Video/test/000053/rgb/000001.png",
"/home/IIT.LOCAL/fceola/workspace/ws_mask/corl-code/python-online-detection/Data/datasets/YCB-Video/test/000059/rgb/000266.png",
"/home/IIT.LOCAL/fceola/workspace/ws_mask/corl-code/python-online-detection/Data/datasets/YCB-Video/test/000056/rgb/000703.png",
"/home/IIT.LOCAL/fceola/workspace/ws_mask/corl-code/python-online-detection/Data/datasets/YCB-Video/test/000048/rgb/001074.png",
"/home/IIT.LOCAL/fceola/workspace/ws_mask/corl-code/python-online-detection/Data/datasets/YCB-Video/test/000049/rgb/002275.png",
"/home/IIT.LOCAL/fceola/workspace/ws_mask/corl-code/python-online-detection/Data/datasets/YCB-Video/test/000054/rgb/000524.png",
"/home/IIT.LOCAL/fceola/workspace/ws_mask/corl-code/python-online-detection/Data/datasets/YCB-Video/test/000056/rgb/000959.png",
"/home/IIT.LOCAL/fceola/workspace/ws_mask/corl-code/python-online-detection/Data/datasets/YCB-Video/test/000050/rgb/001402.png",
"/home/IIT.LOCAL/fceola/workspace/ws_mask/corl-code/python-online-detection/Data/datasets/YCB-Video/test/000058/rgb/000201.png",
"/home/IIT.LOCAL/fceola/workspace/ws_mask/corl-code/python-online-detection/Data/datasets/YCB-Video/test/000057/rgb/001066.png",
"/home/IIT.LOCAL/fceola/workspace/ws_mask/corl-code/python-online-detection/Data/datasets/YCB-Video/test/000054/rgb/000146.png",
"/home/IIT.LOCAL/fceola/workspace/ws_mask/corl-code/python-online-detection/Data/datasets/YCB-Video/test/000050/rgb/000850.png"]
"""
images = ["/home/iiticublap205/IIT/repos/python-online-detection/Data/datasets/YCB-Video/test/000055/rgb/000588.png"]

for i in range(len(images)):
    image = cv2.imread(images[i], 1)
    predictions = coco_demo.run_on_opencv_image(image)
    if not os.path.exists('test_masks_demo'):
        os.makedirs('test_masks_demo')
    cv2.imwrite('test_masks_demo/{}.jpg'.format(i), predictions)
quit()
# load image and then run prediction
if run_single_image:
    #image_path = '/home/iiticublap205/IIT/datasets/iCWT/TABLE-TOP-single-object-masks/test/Images/mug3/00000044.jpg'
    image_path = '/home/IIT.LOCAL/fceola/workspace/ws_papers_repos/YCB-Video/test/000056/rgb/000712.png'

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

