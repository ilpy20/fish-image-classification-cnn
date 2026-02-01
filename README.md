# Fish Image Classification and Localization (CNN)

This project is an учебный проект (Moscow, 2017) on image classification and localization using a convolutional neural network to detect fish in photos. The solution applies a sliding-window crop over images, runs inference with an Inception-based model, and saves detected fish patches for further training and evaluation.

## Overview

- Uses TensorFlow (C++ API) and CImg for image loading, cropping, and inference.
- Prepares a training set by automatically extracting fish-containing patches.
- Retrains the top layer of a pre-trained Inception model to recognize fish categories.
- Processes test images with the same sliding-window approach to localize fish.

## Source Code

- `src/main.cpp` — C++ implementation of the preprocessing, inference, and detection pipeline.

## Build

This is a C++ TensorFlow + CImg program. You need the TensorFlow C++ library, its headers, and CImg (header-only) available on your system.

Example (adjust include/lib paths for your environment):

```sh
clang++ -std=c++11 -O2 -I/path/to/tensorflow/include -I/path/to/CImg \
  src/main.cpp -L/path/to/tensorflow/lib -ltensorflow_cc -ltensorflow_framework \
  -lpthread -ldl -o fish_detector
```

## Run

The program expects a TensorFlow graph and labels file (defaults to the Inception example paths) and an image directory:

```sh
./fish_detector \
  --imagedir /path/to/images \
  --graph /path/to/tensorflow_inception_graph.pb \
  --labels /path/to/imagenet_comp_graph_label_strings.txt
```

## Notes

- The original report and presentation files are intentionally excluded from git.
