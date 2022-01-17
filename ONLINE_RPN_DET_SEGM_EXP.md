You can run the main experiments reported in the work `Learn Fast, Segment Well: Fast Object Segmentation Learning on the iCub Robot` by following the reported instructions.

### Code installation
Please follow the instructions for installations reported at this [link](https://github.com/robotology/online-detection/blob/master/INSTALLATION_GUIDE.md).

### Datasets Download

Depending on the experiment that you want to run, you need to download the required dataset. Namely, if you want to run the experiment on **YCB-Video**, you need to download such dataset. To have data in a standard format which is employed also by other robotics datasets, we download it from the BOP challenge [website](https://bop.felk.cvut.cz/datasets/).

```
cd $HOME_DIR/Data/datasets/YCB-Video/test
wget http://ptak.felk.cvut.cz/6DB/public/bop_datasets/ycbv_test_all.zip
unzip ycbv_test_all.zip && mv test/* . && rmdir test && rm ycbv_test_all.zip

cd $HOME_DIR/Data/datasets/YCB-Video/train_real
wget http://ptak.felk.cvut.cz/6DB/public/bop_datasets/ycbv_train_real.zip
unzip ycbv_train_real.zip && mv train_real/* . && rmdir train_real && rm ycbv_train_real.zip
```
Similarly, if you want to run the experiment on **HO-3D**, you need to create the folder `$HOME_DIR/Data/datasets/HO3D_V2` and download the files *HO3D_v2.zip* and *HO3D_v2_segmentations_rendered.zip* in it, following instructions for *Dataset (version 2)* from [here](https://www.tugraz.at/institute/icg/research/team-lepetit/research-projects/hand-object-3d-pose-annotation/). Then, you need to follow the subsequent instructions. 

```
cd $HOME_DIR/Data/datasets/HO3D_V2
unzip HO3D_v2.zip
rm HO3D_v2.zip
unzip HO3D_v2_segmentations_rendered.zip
rm HO3D_v2_segmentations_rendered.zip
cd $HOME_DIR/src
python ho3d-to-icwt-format.py
rm -rf $HOME_DIR/Data/datasets/HO3D_V2
```

Note: at the end of the installation, remember to `unset HOME_DIR`

## Usage
By modifying the configuration files in the `experiments/configs` folder and by substituting the files with the proper ones for your data, this code allows you to run customized experiments. However, in this repository we provide the configuration files and scripts that are required to reproduce the main experiments in the presented paper. In the following commands, you need to replace *\<dataset\>* with *ycbv*, if you want to run the experiments on the YCB-Video dataset, or with *ho3d*, if you want to use HO-3D. Note that, if you do not specify the paths of the configuration files, the ones for YCB-Video are used by default. All the commands reported below must be run in the `experiments` folder. For all the experiments, we suggest to set a directory to store the output files with the parameter `--output_dir`.
 
To reproduce results of **Ours** you have to run the command:

```
python run_experiment_online_rpn_ood_oos.py --config_file_feature_extraction config_feature_extraction_online_rpn_det_segm_<dataset>.yaml --config_file_online_rpn_detection_segmentation config_online_rpn_detection_segmentation_<dataset>.yaml
```

To reproduce results of **Ours Serial** you have to run the command: 

```
python run_experiment_online_rpn_ood_oos_serial.py --config_file_feature_extraction config_feature_extraction_online_rpn_det_segm_<dataset>_serial.yaml --config_file_rpn config_rpn_<dataset>.yaml --config_file_online_rpn_detection_segmentation config_online_rpn_detection_segmentation_<dataset>_serial.yaml
```

To reproduce results of **O-OS** you have to run the command: 

```
python run_experiment_segmentation.py --config_file_feature_extraction config_feature_extraction_segmentation_<dataset>_t_ro.yaml --config_file_online_detection_online_segmentation config_online_detection_segmentation_<dataset>_t_ro.yaml
```

To train the *Mask R-CNN* baselines you have to run the following commands. Please consider using the parameter `--train_for_time`, if you want to set the total training time as in the experiments in Sec. VII of the paper.

To reproduce results of **Mask R-CNN (full)** you have to run the command: 

```
python run_experiment_full_train.py --config_file configs/config_full_train_<dataset>.yaml
```

To reproduce results of **Mask R-CNN (output layers)** you have to run the command:

```
python run_experiment_fine_tuning.py --config_file configs/config_fine_tuning_<dataset>.yaml --fine_tune_RPN
```

To reproduce results of **Mask R-CNN (store features)** you have to run the following command. The first time that you run this experiment, you need to set also the parameters `--config_file_feature_extraction` and `--extract_backbone_features` for backbone features computation.

```
python run_experiment_fine_tuning.py --config_file configs/config_fine_tuning_<dataset>_from_feat.yaml --fine_tune_RPN --use_backbone_features
```

If you want to visualize qualitative results, you need to use the script `visualize_masks_online_segmentation.py`. To visualize results using models obtained with **Ours**, **Ours Serial** or **O-OS** you need to specify the directory where the models are stored with the parameter `--models_dir`. Instead, if you want to visualize results for the *Mask R-CNN* baselines, you need to use the parameter `--mask_rcnn_model`.  In both cases, you need to use the parameter `--dataset <dataset>` to specify the dataset used for training. 

**Important**: if you have more than one GPU available, before running an experiment, you have to set the number of the GPU that you want to use (only one) with the command `export CUDA_VISIBLE_DEVICES=number_of_the_gpu`.

