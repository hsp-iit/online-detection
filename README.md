

# On-line Object Detection

This repository contains the source code for the experiments carried out for the On-line Object Detection project.

## Abstract
Object detection is a fundamental ability for robots interacting within an environment. While stunningly effective, state-of-the-art deep learning methods require huge amounts of labeled images and hours of training which does not favour such scenarios. In this project, we aim at designing algorithmic solutions to alleviate these requirements for this task, while preserving the state-of-the-art precision and reliability.
The proposed methods are typically validated on both computer vision and robotics datasets. This repository allows to reproduce the main experiments of the proposed works and allows the user to test the pipeline with other datasets.

## Description

*Pipeline picture*

This pcture presents the current architecture of the pipeline. The Feature Extraction Module relies on Mask R-CNN architecture and the proposed On-line RPN, to extract deep features and predict RoIs from each input image. The On-line Detection Module performs RoIs classification and refinement, providing as output the detections for the input image. The green blocks are trained off-line on the FEATURE-TASK, while the yellow blocks are trained on-line on the TARGET-TASK.

## Installation guide
You can find the instructions for installation at this [link]().

## Experiments

### Fast Region Proposal Learning for Object Detection for Robotics
You can find the instructions to replicate experiments at this [link]()

## References
If you use this code you can cite the following works:

```
@INPROCEEDINGS{maiettini2018,
	author={E. Maiettini and G. Pasquale and L. Rosasco and L. Natale},
	booktitle={2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
	title={Speeding-up Object Detection Training for Robotics with FALKON},
	year={2018},
	month={Oct},}
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

