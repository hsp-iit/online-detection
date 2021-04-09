You can run the main experiments reported in the work `Fast Object Segmentation Learning with Kernel-based Methods for
Robotics` by following the reported instructions.

### Code installation
Please follow the instructions for installations reported at this [link](https://github.com/robotology/online-detection/blob/master/INSTALLATION_GUIDE.md).

### Datasets Download

To reproduce the experiments of the paper, you need to download the YCB-Video dataset. In order to have data in a standard format which is employed also by other robotic dataset, we download it from the BOP challenge [website](https://bop.felk.cvut.cz/datasets/).

```
cd $HOME_DIR/Data/datasets/YCB-Video/test
wget http://ptak.felk.cvut.cz/6DB/public/bop_datasets/ycbv_test_all.zip
unzip ycbv_test_all.zip && mv test/* . && rmdir test && rm ycbv_test_all.zip

cd $HOME_DIR/Data/datasets/YCB-Video/train_real
wget http://ptak.felk.cvut.cz/6DB/public/bop_datasets/ycbv_train_real.zip
unzip ycbv_train_real.zip && mv train_real/* . && rmdir train_real && rm ycbv_train_real.zip
```


Note: at the end of the installation, remember to `unset HOME_DIR`

## Usage
By modifying the configuration files in the `experiments/configs` folder and by substituting the files with the proper ones for your data, this code allows you to run customized experiments. However, in this repository we provide you with the  configuration files and scripts that are required to reproduce the main experiment in the presented paper. To do this, you have to run the script `experiments/run_experiment_segmentation.py` and to properly set command line arguments. Some examples will be provided below. For additional information please refer to the helper of the script, running the command `python run_experiment_segmentation.py -h` in the  `experiments` directory.

In the `experiments/configs` you can find two categories of configuration files related to this paper:
 - config_feature_extraction_segmentation_ycbv.yaml: sets parameters for feature extraction of **Ours** experiment, reported in the second row of *Table 1* in the paper.
 - config_online_detection_segmentation_ycbv.yaml: sets parameters to train online detection and online segmentation of **Ours** experiment, reported in the second row of *Table 1* in the paper.

**Important**: if you have more than one GPU available, before running an experiment, you have to set the number of the GPU that you want to use (only one) with the command `export CUDA_VISIBLE_DEVICES=number_of_the_gpu`


### YCB-Video with COCO FEATURE-TASK experiment
To reproduce the results of **Ours** experiment in the second row of *Table 1*, in the `experiments` folder you have to run the command

`python run_experiment_segmentation.py`.

If you do not set from command line an output directory, experiment's results will be saved in the `experiments/online_segmentation_experiment_ycbv/result.txt` file.

