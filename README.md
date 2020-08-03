# Fast Region Proposal Learning for Object Detection for Robotics

This repository contains the source code used to run the experiments of the paper _Fast Region Proposal Learning for Object Detection for Robotics_.

## Abstract
Object detection is a fundamental task for robots to operate in unstructured environments. Today, there are several deep learning algorithms that solve this task with remarkable performance. Unfortunately, training such systems requires several hours of GPU time. For robots, to successfully adapt to changes in the environment or learning new objects, it is also important that object detectors can be re-trained in a short amount of time. A recent method [1] proposes an architecture that leverages on the powerful representation of deep learning descriptors, while permitting fast adaptation time. Leveraging on the natural decomposition of the task in (i) regions candidate generation, (ii) feature extraction and (iii) regions classification, this method performs fast adaptation of the detector, by only re-training the classification layer. This shortens training time while maintaining state-of-the-art performance. In this paper, we firstly demonstrate that a further boost in accuracy can be obtained by adapting, in addition, the regions candidate generation on the task at hand. Secondly, we extend the object detection system presented in [1] with the proposed fast learning approach, showing experimental evidence on the improvement provided in terms of speed and accuracy on two different robotics datasets.

## Installation's Guide
The following instructions will guide you in preparing your system before running the experiments proposed in the paper.

### Software Requirements
- NVIDIA's cuda *version*
- Python 3.6
- ... #FEDERICO

### Required Python Packages
```
pip install ninja==1.9.0.post1 yacs==0.1.7 cython==0.29.17 matplotlib==3.2.1 
pip install tqdm==4.45.0 opencv-python==4.2.0.34
pip install torchvision==0.5.0
pip install scipy==1.4.1
pip install h5py==2.10.0
```

### Source Code Preparation
```
unzip instructions #FEDERICO
```

### Mask R-CNN Installation
One of the dependencies of this code is the source code of Mask R-CNN [2]. The following instructions will guide you in the installation of mask rcnn's repository and  dependencies.

#### Set installation directory
```
mkdir external/mask_rcnn
cd external/mask_rcnn
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
In order to complete the installation of the repository `cityscapesScripts`, after cloning the repository, you may need to change in the `setup.py` file the line 27. Specifically, from `with open("README.md") as f:`  to `open("README.md", encoding='utf-8') as f:` 
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
cd ../..
git clone --recurse-submodules https://github.com/FalkonML/falkon.git
cd falkon
git checkout faster-mmv
git checkout b0e45e5495a4e9801f9bf4608a92fb4a7f95d4df
pip install ./keops
pip install .
```


## Usage
By modifying the configuration files in the `Conf` folder and by substituting the files `name_file` with the proper ones for your data, this code allows you to run customized experiments. However, in this repository we provide you with the  configuration files and scripts that are required to reproduce the main experiments in the presented paper.

### iCWT: different objects, same setting.
#FEDERICO

### TABLE-TOP: different objects and setting.
#FEDERICO

## Statement of Contribution
The code contained in this repository is part of the contribution of the presented work. Specifically, we produced all the code except for the repositories contained in the `external` folder. Please, also note that the code contained in `folder_name #FEDERICO`  contains a modified version of some Mask R-CNN's functions that we changed for the implementation of the `On-line RPN` and for feature extraction.


## References
[1] E. Maiettini, G. Pasquale, L. Rosasco, and L. Natale. On-line object detection: a robotics challenge. Autonomous Robots, Nov 2019. ISSN 1573-7527. doi:10.1007/s10514-019-09894-9. URL https://doi.org/10.1007/s10514-019-09894-9.

[2] K. He, G. Gkioxari, P. Dollár, and R. B. Girshick. Mask r-cnn. 2017 IEEE International Conference on Computer Vision (ICCV), pages 2980–2988, 2017.

[3] A. Rudi, L. Carratino, and L. Rosasco. Falkon: An optimal large scale kernel method. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems 30, pages 3888–3898. Curran Associates, Inc., 2017.
