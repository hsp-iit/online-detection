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
To start source code preparation, please, clone this repository as follows:
```
git clone https://github.com/robotology/online-detection
cd online-detection
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
git checkout --recurse-submodules 801e5f3d01b9ec5b3142f6376b18f18377c0dd37
pip install ./keops
pip install .
```

