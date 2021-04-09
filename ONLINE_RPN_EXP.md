You can run the main experiments reported in the work `Fast Region Proposal Learning for Object Detection for Robotics` by following the reported instructions.

### Code installation
Please follow the instructions for installations reported at this [link](https://github.com/robotology/online-detection/blob/master/INSTALLATION_GUIDE.md).

### Datasets Download

Depending on the experiment that you want to run, you need to download the required dataset. Namely, if you want to run the experiment on the **30 objects identification task from the iCubWorld Transformations dataset (iCWT)**, you need to download such dataset, following the subsequent instructions.
```
cd $HOME_DIR/Data/datasets/iCWT/iCubWorld-Transformations
mkdir Images
cd Images
wget https://zenodo.org/record/835510/files/part1.tar.gz
tar -xvf part1.tar.gz --strip 1
wget https://zenodo.org/record/835510/files/part2.tar.gz
tar -xvf part2.tar.gz --strip 1
wget https://zenodo.org/record/835510/files/part3.tar.gz
tar -xvf part3.tar.gz --strip 1
wget https://zenodo.org/record/835510/files/part4.tar.gz
tar -xvf part4.tar.gz --strip 1

rm part1.tar.gz
rm part2.tar.gz
rm part3.tar.gz
rm part4.tar.gz

cd $HOME_DIR/Data/datasets/iCWT/iCubWorld-Transformations
mkdir Annotations
cd Annotations
wget https://zenodo.org/record/1227305/files/Annotations_refined.tar.gz
tar -xvf Annotations_refined.tar.gz --strip 1
rm Annotations_refined.tar.gz

cd $HOME_DIR/Data/datasets/iCWT/iCubWorld-Transformations_manual
ln -s $HOME_DIR/Data/datasets/iCWT/iCubWorld-Transformations/Images .
mkdir Annotations
cd Annotations
wget https://zenodo.org/record/2563223/files/Annotations_manual.tar.gz
tar -xvf Annotations_manual.tar.gz --strip 1
rm Annotations_manual.tar.gz
```
Similarly, if you want to run the experiment on the **21 objects identification task from the TABLE-TOP dataset**, you need to download such dataset, following the subsequent instructions.

```
cd $HOME_DIR/Data/datasets/iCWT/TABLE-TOP
mkdir Annotations
cd Annotations
wget https://zenodo.org/record/3970624/files/table_top_annotations.tar.xz
tar -xvf table_top_annotations.tar.xz --strip 1
rm table_top_annotations.tar.xz

cd $HOME_DIR/Data/datasets/iCWT/TABLE-TOP
mkdir Images
cd Images
wget https://zenodo.org/record/3970624/files/table_top_images.tar.xz
tar -xvf table_top_images.tar.xz --strip 1
rm table_top_images.tar.xz
```

### Pretrained Feature Extractors Download
Create the `pretrained_feature_extractors` directory, where you will download the pretrained feature extractors that you need to run the experiments.

```
cd $HOME_DIR/Data
mkdir pretrained_feature_extractors
cd pretrained_feature_extractors
```

If you want to run the experiment on the **30 objects identification task from the iCubWorld Transformations dataset (iCWT)**, you need to download the feature extractor with the following command.

```
 wget --content-disposition https://istitutoitalianotecnologia-my.sharepoint.com/:u:/g/personal/federico_ceola_iit_it/EW4b1wC905JDgKkrPG6nmmABAlDNqtbiKdnQN7QN0NoO6A?download=1
```

Similarly, if you want to run the experiment on the **21 objects identification task from the TABLE-TOP dataset**, you need to download the feature extractor with the following command.

```
wget --content-disposition https://istitutoitalianotecnologia-my.sharepoint.com/:u:/g/personal/federico_ceola_iit_it/EY1Bu31xC_xCviY0qxdPwKIBBRlQfjSJ_MX1xXs97AVH1A?download=1
```

Note: at the end of the installation, remember to `unset HOME_DIR`

## Usage
By modifying the configuration files in the `experiments/configs` folder and by substituting the files with the proper ones for your data, this code allows you to run customized experiments. However, in this repository we provide you with the  configuration files and scripts that are required to reproduce the main experiments in the presented paper.
For all the experiments that you want to reproduce, you have to run the script `experiments/run_experiment.py` and to properly set command line arguments. Some examples will be provided in the two following subsections. For additional information please refer to the helper of the script, running the command `python run_experiment.py -h` in the  `experiments` directory.

In the `experiments/configs` you can find four categories of configuration files:
 - config_rpn_*experiment_name*.yaml: sets parameters for RPN's feature extraction.
 - config_detector_*experiment_name*.yaml: sets parameters for detector's feature extraction.
 - config_online_detection_*experiment_name*.yaml: sets parameters for the **O-OD** experiment (3rd row of *Table 1* and *Table 2* in the paper).
 - config_online_online_rpn_detection_*experiment_name*.yaml: sets parameters for the **Ours** experiment (5th row of *Table 1* and *Table 2* in the paper).

**Important**: if you have more than one GPU available, before running an experiment, you have to set the number of the GPU that you want to use (only one) with the command `export CUDA_VISIBLE_DEVICES=number_of_the_gpu`


### iCWT: different objects, same setting.
To reproduce the results of the **O-OD** experiment on the **iCWT** dataset reported in the **Table 1** of the paper, in the `experiments` folder you have to run the command

`python run_experiment.py --icwt30 --only_ood`.

To reproduce **Ours** experiment you have to run the command

`python run_experiment.py --icwt30`.

If you do not set from command line an output directory, experiment's results will be saved in the `experiments/icwt30_experiment/result.txt` file.

### TABLE-TOP: different objects and setting.
Similarly to the iCWT experiment, to reproduce the results of the **O-OD** experiment on the **TABLE-TOP** dataset reported in the **Table 2** of the paper, in the `experiments` folder you have to run the command

`python run_experiment.py --only_ood`.

To reproduce **Ours** experiment you have to run the command

`python run_experiment.py`.

If you do not set from command line an output directory, experiment's results will be saved in the `experiments/tabletop_experiment/result.txt` file.

