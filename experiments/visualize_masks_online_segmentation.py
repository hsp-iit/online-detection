import sys
import os
basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'region-classifier')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'region-refiner')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'feature-extractor')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'accuracy-evaluator')))

from mrcnn_modified.config import cfg
from mrcnn_modified.demo.predictor_online_segmentation import OnlineSegmentationDemo
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', action='store', type=str, default="config_segmentation_visualizer.yaml", help='Manually set configuration file, by default it is config_segmentation_ycb_demo.yaml. If the specified path is not absolute, the config file will be searched in the experiments/configs directory')
parser.add_argument('--image_path', action='store', type=str, default="Data/datasets/YCB-Video/test/000055/rgb/000588.png", help='Set image to be processed. If you do not pass an absolute path, it must be relative to the python-online-detection path')
parser.add_argument('--confidence_threshold', action='store', type=float, default=0.2, help='Set detection score threshold')
parser.add_argument('--images_paths_from_file', action='store_true', help='Read images paths from file')
parser.add_argument('--list_path', action='store', type=str, default="src/modules/feature-extractor/mrcnn_modified/demo/test_images.txt", help='Run the experiment on a list of images reported in file at the given path. If you do not pass an absolute path, it must be relative to the python-online-detection path')
parser.add_argument('--do_not_display_images', action='store_true', help='Run demo experiment, but do not display the images')
parser.add_argument('--write_outputs', action='store_true', help='Write outputs on disk')
parser.add_argument('--output_dir', action='store', type=str, default='test_masks_oos', help='Set where images will be saved, if the write_outputs option is True. Relative paths are relative to the experiments folder')
parser.add_argument('--models_dir', action='store', type=str, help='Specify where models for online RPN, detection or segmentation are. Relative paths are relative to the experiments folder')
parser.add_argument('--dataset', action='store', type=str, default='ycbv', help='Specify dataset to overlay on the image correct classes names. For the iCWT TARGET-TASK, the argument must be iCWT_TT. For YCB-Video ycbv')
parser.add_argument('--fill_masks', action='store_true', help='Set if masks must be filled in the visualization')
parser.add_argument('--save_png_masks', action='store_true', help='Set if predicted masks must be saved on disk')
parser.add_argument('--mask_rcnn_model', action='store_true', help='Set if running the demo on a fine-tuned or fully trained model of Mask R-CNN')



args = parser.parse_args()

if args.config_file.startswith("/"):
    config_file = args.config_file
else:
    config_file = os.path.abspath(os.path.join(basedir, "configs", args.config_file))

if args.models_dir:
    if args.models_dir.startswith("/"):
        models_dir = args.models_dir
    else:
        models_dir = os.path.abspath(os.path.join(basedir, args.models_dir))
else:
    models_dir = args.models_dir

if args.write_outputs:
    if args.output_dir.startswith("/"):
        output_dir = args.output_dir
    else:
        output_dir = os.path.abspath(os.path.join(basedir, args.output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

cfg.merge_from_file(config_file)

if args.dataset:
    dataset = args.dataset
else:
    dataset = None
demo = OnlineSegmentationDemo(
    cfg,
    confidence_threshold=args.confidence_threshold,
    models_dir=models_dir,
    dataset=dataset,
    fill_masks=args.fill_masks,
    mask_rcnn_model=args.mask_rcnn_model
)

images_paths = []
if args.images_paths_from_file:
    if args.list_path.startswith("/"):
        list_path = args.list_path
    else:
        list_path = os.path.abspath(os.path.join(basedir, os.path.pardir, args.list_path))
    f = open(list_path)
    for line in f:
        images_paths.append(line.rstrip('\n'))
    f.close()
else:
    images_paths.append(args.image_path)

# load images and compute prediction
for i in range(len(images_paths)):
    if not images_paths[i].startswith("/"):
        images_paths[i] = os.path.abspath(os.path.join(basedir, os.path.pardir, images_paths[i]))
    image = cv2.imread(images_paths[i], 1)
    predictions_with_masks_values = demo.run_on_opencv_image(image)
    predictions = predictions_with_masks_values[0]
    if not args.do_not_display_images:
        cv2.imshow('Predictions', predictions)
        cv2.waitKey(0)
    if args.write_outputs:
        cv2.imwrite(os.path.join(output_dir, str(format(i,'05d'))+'_'+os.path.splitext(os.path.basename(images_paths[i]))[0])+'.jpg', predictions)
    if args.save_png_masks:
        if predictions_with_masks_values is not None:
            # Keep track of the instances of the same class
            labels_ids_set = set(predictions_with_masks_values[2])
            dict_of_ids_instances = {i: 0 for i in labels_ids_set}
            for mask, label in zip(predictions_with_masks_values[1], predictions_with_masks_values[2]):
                png_mask_file_name = os.path.join(output_dir, os.path.splitext(os.path.basename(images_paths[i]))[0]+'_'+label+'_'+str(dict_of_ids_instances[label])+'.png')
                dict_of_ids_instances[label] += 1
                mask=mask.astype(np.uint8).squeeze()*255
                cv2.imwrite(png_mask_file_name, mask)
cv2.destroyAllWindows()
