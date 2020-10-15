

# Fast Region Proposal Learning for Object Detection for Robotics

This repository contains the source code used to run the experiments of the paper _Fast Region Proposal Learning for Object Detection for Robotics_.

## Abstract
Object detection is a fundamental task for robots to operate in unstructured environments. Today, there are several deep learning algorithms that solve this task with remarkable performance. Unfortunately, training such systems requires several hours of GPU time. For robots, to successfully adapt to changes in the environment or learning new objects, it is also important that object detectors can be re-trained in a short amount of time. A recent method [1] proposes an architecture that leverages on the powerful representation of deep learning descriptors, while permitting fast adaptation time. Leveraging on the natural decomposition of the task in (i) regions candidate generation, (ii) feature extraction and (iii) regions classification, this method performs fast adaptation of the detector, by only re-training the classification layer. This shortens training time while maintaining state-of-the-art performance. In this paper, we firstly demonstrate that a further boost in accuracy can be obtained by adapting, in addition, the regions candidate generation on the task at hand. Secondly, we extend the object detection system presented in [1] with the proposed fast learning approach, showing experimental evidence on the improvement provided in terms of speed and accuracy on two different robotics datasets.

## Installation's Guide
The following instructions will guide you in preparing your system before running the experiments proposed in the paper.

### Software Requirements
- NVIDIA's cuda *version* 10.1
- Python 3.6.9

### Required Python Packages
```
pip install ninja==1.9.0.post1 yacs==0.1.7 cython==0.29.17 matplotlib==3.2.1 
pip install tqdm==4.45.0 opencv-python==4.2.0.34
pip install torchvision==0.5.0
pip install scipy==1.4.1
```

### Source Code Preparation
To start source code preparation, copy `python-online-detection.zip` in the folder where you want to place the code and from that folder execute the following commands.
```
unzip python-online-detection.zip
cd python-online-detection
export HOME_DIR=$PWD
```

### Mask R-CNN Installation
One of the dependencies of this code is the source code of Mask R-CNN [2]. The following instructions will guide you in the installation of mask rcnn's repository and  dependencies.

#### Set installation directory
```
cd $HOME_DIR/external
mkdir mask_rcnn
cd mask_rcnn
export INSTALL_DIR=$PWD
```
#### Install pycocotools
```
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
```
#### Install cityscapesScripts
In order to complete the installation of the repository `cityscapesScripts`, after cloning the repository, you may need to change in the `setup.py` file the line 27. Specifically, from `with open("README.md") as f:`  to `with open("README.md", encoding='utf-8') as f:` 
```
cd $INSTALL_DIR
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts/
python setup.py build_ext install
```

#### Install apex
```
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
```
#### Install maskrcnn-benchmark
```
cd $INSTALL_DIR
git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
cd maskrcnn-benchmark
python setup.py build develop
```
Note: at the end of the installation of Mask R-CNN source code, remember to `unset INSTALL_DIR` .

### FALKON
One of the dependencies of this code is the repository of FALKON [3]. The following instructions will guide you in the installation of the source code. 
```
cd $HOME_DIR/external
git clone --recurse-submodules https://github.com/FalkonML/falkon.git
cd falkon
pip install ./keops
pip install .
```

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


## Statement of Contribution
The code contained in this repository is part of the contribution of the presented work. Specifically, we produced all the code except for the repositories contained in the  `external` folder. Please, also note that the code contained in `src/modules/feature-extractor/mrcnn_modified`  contains a modified version of some Mask R-CNN's functions that we changed for the implementation of the `On-line RPN` and for feature extraction.


## References
[1] E. Maiettini, G. Pasquale, L. Rosasco, and L. Natale. On-line object detection: a robotics challenge. Autonomous Robots, Nov 2019. ISSN 1573-7527. doi:10.1007/s10514-019-09894-9. URL https://doi.org/10.1007/s10514-019-09894-9.

[2] K. He, G. Gkioxari, P. Dollár, and R. B. Girshick. Mask r-cnn. 2017 IEEE International Conference on Computer Vision (ICCV), pages 2980–2988, 2017.

[3] A. Rudi, L. Carratino, and L. Rosasco. Falkon: An optimal large scale kernel method. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems 30, pages 3888–3898. Curran Associates, Inc., 2017.




