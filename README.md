# Masked Face Detector

Welcome to the Masked Face Detector GitHub repository! This project aims to detect masked faces using two different models: one based on ResNet50V2 and the other on MobileNetV2. The ResNet50V2 model offers higher accuracy but slower inference times, while the MobileNetV2 model is faster but slightly less accurate.

## Introduction
Detecting whether a face is masked or not is a critical task for various applications, including public health monitoring, security, and access control. This repository provides two models to help you achieve this task efficiently.

## Models
1. ResNet50V2 Model
Accuracy: High
Inference Speed: Slower

2. MobileNetV2 Model
Accuracy: Slightly lower than ResNet50V2
Inference Speed: Faste

### ResNet Folder
In the ResNet folder, you'll find the following files and resources:

- ResNet-CrossValidation.xlsx: This spreadsheet shows the cross-validation results of the ResNet50V2 model.

- ResNet50V2_model: This directory contains the ResNet50V2 model itself.

- cross-validation.py: Use this script to perform cross-validation experiments on the ResNet50V2 model.

- resnet-epochs.png: An image representing the training epochs during model training.

- train-model.py: This script contains the algorithm to train the ResNet50V2 model.

## MobileNet Folder
In the MobileNet folder, you'll find similar resources and files as in the ResNet folder, but they are specific to the MobileNetV2 model.

## Face Detector Folder
The face_detector folder contains the model architecture and weights required for face detection. Additionally, you'll find the detect_mask_image.py file, which is the algorithm used for making mask detection on input images.

## Contributing
We welcome contributions to this project

## License
This project is licensed under the MIT License.
