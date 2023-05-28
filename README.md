# DLAV

## Overview

This repository was made for the Deep Learning for Autonomous Vehicles course at EPFL. The task of this project is 3D Human Pose Estimation from a single RGB camera as implemented in the paper "Camera Distance-aware Top-down Approach for 3D Multi-person Pose Estimation from a Single RGB Image" (https://arxiv.org/pdf/1907.11346v2.pdf). 

The model uses three networks for this task. The first one is called DetecNet and is used to compute the bounding boxes around each person in the frame. They are then fed to the second network called RootNet that estimates the absolute position of the root of each person. In parallel, the third network called PoseNet uses the bounding boxes to estimate the relative 3D position of all the joints for every subject. We can then combine them to obtain the absolute position for all people in the frame.

## Experiments

## Dataset

We have used the dataset Human 3.6M for this task as it provides a very large amount of data and 3D annotations. As it is too large to be handled on a regular computer, we used scitas to access the data.

## Training

For training or testing, you must download annotations.zip from Google Drive (you can delete everything after the .zip extension for extraction): https://drive.google.com/drive/folders/1r0B9I3XxIIW_jsXjYinDpL6NFcxTZart

The .json files must be put in a folder named "annotations" in the same directory as Human36M.py. These files contain the annotations and the ground truth bounding boxes for the Human3.6M dataset.

## Inference

## Results

## Conclusion

## References

@InProceedings{Moon_2019_ICCV_3DMPPE,
author = {Moon, Gyeongsik and Chang, Juyong and Lee, Kyoung Mu},
title = {Camera Distance-aware Top-down Approach for 3D Multi-person Pose Estimation from a Single RGB Image},
booktitle = {The IEEE Conference on International Conference on Computer Vision (ICCV)},
year = {2019}
}
