import os
import sys
import argparse

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'feature-extractor')))

from feature_extractor import FeatureExtractor

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', action='store', type=str, default='fine_tuning_ycbv', help='Set experiment\'s output directory.')
parser.add_argument('--config_file', action='store', type=str, default="configs/config_fine_tuning_ycbv.yaml", help='Manually set configuration file, by default it is configs/config_full_train_ycbv.yaml. If the specified path is not absolute, the config file will be searched in the experiments directory')
parser.add_argument('--fine_tune_RPN', action='store_true', help='Fine-tune also last RPN layers')
parser.add_argument('--use_backbone_features', action='store_true', help='Load features extracted with the backbone instead of the images as input')


args = parser.parse_args()

# Set and create output directory
if args.output_dir:
    if args.output_dir.startswith('/'):
        output_dir = args.output_dir
    else:
        output_dir = os.path.abspath(os.path.join(basedir, args.output_dir))

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

if args.config_file.startswith("/"):
    cfg_feature_task = args.config_file
else:
    cfg_feature_task = os.path.abspath(os.path.join(basedir, args.config_file))

# Initialize feature extractor
feature_extractor = FeatureExtractor(cfg_path_feature_task=cfg_feature_task)

feature_extractor.trainFeatureExtractor(output_dir=output_dir, fine_tune_last_layers=True, fine_tune_rpn=args.fine_tune_RPN, use_backbone_features=args.use_backbone_features)
