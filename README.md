

# On-line Object Detection and Instance Segmentation

This repository contains the python version of the source code for the experiments carried out for the *On-line Object Detection and Instance Segmentation* project.

## Abstract
Object detection and instance segmentation are fundamental tasks for robots interacting within an environment. While stunningly effective, state-of-the-art deep learning methods require huge amounts of labeled images and long training sessions which does not favour such scenarios. In this project, we aim at designing algorithmic solutions to alleviate these requirements for this task, while preserving the state-of-the-art precision and reliability.
The proposed methods are typically validated on both computer vision and robotics datasets. This repository allows to reproduce the main experiments of the proposed works and allows the user to test the pipeline with other datasets.

## Description

![pipeline_orpn_oos_reviewed_2_compressed](https://user-images.githubusercontent.com/32268209/149495420-b84dfdcf-8263-4be3-b6a3-9e1dd7f7b3b5.png)

This picture presents the current architecture of the pipeline. The *Feature Extraction Module* relies on the *Mask R-CNN* architecture and the proposed *On-line RPN*, to extract deep features and predict *RoIs* from each input image. The *On-line Detection Module* performs *RoIs* classification and refinement, providing as output the detections for the input image. The green blocks are trained off-line on the FEATURE-TASK, while the yellow blocks are trained on-line on the TARGET-TASK.

## Installation guide
You can find the instructions for installation at this [link](https://github.com/hsp-iit/online-detection/blob/master/INSTALLATION_GUIDE.md).

## Experiments
We provide the links to instructions to reproduce the main experiments of the presented works.

### Learn Fast, Segment Well: Fast Object Segmentation Learning on the iCub Robot
You can find the instructions to replicate experiments at this [link](https://github.com/hsp-iit/online-detection/blob/master/ONLINE_RPN_DET_SEGM_EXP.md).

### Fast Object Segmentation Learning with Kernel-based Methods for Robotics
You can find the instructions to replicate experiments at this [link](https://github.com/hsp-iit/online-detection/blob/master/ONLINE_SEGMENTATION_EXP.md).

### Fast Region Proposal Learning for Object Detection for Robotics
You can find the instructions to replicate experiments at this [link](https://github.com/hsp-iit/online-detection/blob/master/ONLINE_RPN_EXP.md).

## References
If you use this code, please, cite the following works:

```
@INPROCEEDINGS{ceola2021oos,
  author={Ceola, Federico and Maiettini, Elisa and Pasquale, Giulia and Rosasco, Lorenzo and Natale, Lorenzo},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={Fast Object Segmentation Learning with Kernel-based Methods for Robotics}, 
  year={2021},
  volume={},
  number={},
  pages={13581-13588},
  doi={10.1109/ICRA48506.2021.9561758}
}
```
```
@Article{maiettini2019a,
	author="Maiettini, Elisa
	and Pasquale, Giulia
	and Rosasco, Lorenzo
	and Natale, Lorenzo",
	title="On-line object detection: a robotics challenge",
	journal="Autonomous Robots",
	year="2019",
	month="Nov",
	day="25",
	issn="1573-7527",
	doi="10.1007/s10514-019-09894-9",
	url="https://doi.org/10.1007/s10514-019-09894-9"
}
```
```
@INPROCEEDINGS{maiettini2018,
	author={E. Maiettini and G. Pasquale and L. Rosasco and L. Natale},
	booktitle={2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
	title={Speeding-up Object Detection Training for Robotics with FALKON},
	year={2018},
	month={Oct},
}
```
```
@article{ceola2020rpn,
  title={Fast region proposal learning for object detection for robotics},
  author={Ceola, Federico and Maiettini, Elisa and Pasquale, Giulia and Rosasco, Lorenzo and Natale, Lorenzo},
  journal={arXiv preprint arXiv:2011.12790},
  year={2020}
}
```
