# Fish Image Classification and Localization (CNN)

## Project Summary

This repository contains a 2017 student project that detects and localizes fish in images using a convolutional neural network. It applies a sliding-window crop over each image, runs inference with a pre-trained Inception-based TensorFlow model, and saves high-confidence fish crops into per-species `/detected` folders. The detected crops can be used to retrain the top layer of the model for improved category recognition, and the same pipeline can be used to localize fish in test images.

This project is a student project (Moscow, 2017) on image classification and localization using a convolutional neural network to detect fish in photos. The solution applies a sliding-window crop over images, runs inference with an Inception-based model, and saves detected fish patches for further training and evaluation.

## Overview

- Uses TensorFlow (C++ API) and CImg for image loading, cropping, and inference.
- Prepares a training set by automatically extracting fish-containing patches.
- Retrains the top layer of a pre-trained Inception model to recognize fish categories.
- Processes test images with the same sliding-window approach to localize fish.

## Data Source

- Kaggle: *The Nature Conservancy Fisheries Monitoring* (used for training/test images).

## Method (from the report)

1. **Training data preparation**: apply a sliding window over raw images, classify each crop with a pre-trained Inception model, and save fish-containing crops into a `/detected` subfolder.
2. **Training**: retrain the top layer of Inception on the detected crops (per fish category), producing an output graph (e.g., `output_graph.pb`).
3. **Testing**: run the same sliding-window + classification pipeline on test images to localize fish.

## Model / Files

- Default graph path in code: `tensorflow/examples/label_image/data/tensorflow_inception_graph.pb`
- Default labels path in code: `tensorflow/examples/label_image/data/imagenet_comp_graph_label_strings.txt`
- Trained output graph mentioned in report: `output_graph.pb`
- Detected crop output: `/detected` folder under each species directory

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
- The report describes development on MacOS with XCode; adjust build steps for your environment.
