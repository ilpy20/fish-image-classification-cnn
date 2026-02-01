# Fish Image Classification and Localization (CNN)

This project is an учебный проект (Moscow, 2017) on image classification and localization using a convolutional neural network to detect fish in photos. The solution applies a sliding-window crop over images, runs inference with an Inception-based model, and saves detected fish patches for further training and evaluation.

## Overview

- Uses TensorFlow (C++ API) and CImg for image loading, cropping, and inference.
- Prepares a training set by automatically extracting fish-containing patches.
- Retrains the top layer of a pre-trained Inception model to recognize fish categories.
- Processes test images with the same sliding-window approach to localize fish.

## Source Code

- `src/main.cpp` — C++ implementation of the preprocessing, inference, and detection pipeline.

## Notes

- The original report and presentation files are intentionally excluded from git.
