import os
import sys
import argparse

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src')))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, 'src', 'modules', 'feature-extractor')))

from feature_extractor import FeatureExtractor

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', action='store', type=str, default='full_train_ycbv', help='Set experiment\'s output directory.')
parser.add_argument('--config_file', action='store', type=str, default="configs/config_full_train_ycbv.yaml", help='Manually set configuration file, by default it is configs/config_full_train_ycbv.yaml. If the specified path is not absolute, the config file will be searched in the experiments directory')
parser.add_argument('--train_for_time', action='store', type=str, help='Train mask for the input training time. It must be in the format XXh:YYm:ZZs.')


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

training_seconds=None
if args.train_for_time:
    for i in range(len(args.train_for_time)):
        if (i == 0 or i ==1 or i ==4 or i==5 or i==8 or i ==9):
            if not args.train_for_time[i].isdigit():
                print('The training time format must be XXh:YYm:ZZs.')
                quit()
        if (i == 3 or i ==7):
            if not args.train_for_time[i] == ':':
                print('The training time format must be XXh:YYm:ZZs.')
                quit()
        if i == 2 and not args.train_for_time[i] == 'h':
            print('The training time format must be XXh:YYm:ZZs.')
            quit()
        if i == 6 and not args.train_for_time[i] == 'm':
            print('The training time format must be XXh:YYm:ZZs.')
            quit()
        if i == 10 and not args.train_for_time[i] == 's':
            print('The training time format must be XXh:YYm:ZZs.')
            quit()
    seconds = 10*int(args.train_for_time[8]) + int(args.train_for_time[9])
    minutes = 10*int(args.train_for_time[4]) + int(args.train_for_time[5])
    hours = 10*int(args.train_for_time[0]) + int(args.train_for_time[1])

    training_seconds = seconds + 60*minutes + 3600*hours

# Initialize feature extractor
feature_extractor = FeatureExtractor(cfg_path_feature_task=cfg_feature_task)

feature_extractor.trainFeatureExtractor(output_dir=output_dir, training_seconds=training_seconds)
