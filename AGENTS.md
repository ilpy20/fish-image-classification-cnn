# Repository Guidelines

## Project Structure & Module Organization

- `src/main.cpp` contains the full C++ pipeline for sliding-window inference, crop saving, and label scoring.
- `assets/` holds charts and sample images referenced by the report/presentation.
- Root-level docs: `README.md` describes the approach, build prerequisites, and run flags.
- Generated outputs (from running the detector) land in per-species `/detected` subfolders next to the image data.

## Build, Test, and Development Commands

This repo does not use a build system; compile directly with your local TensorFlow and CImg installs.

```sh
clang++ -std=c++11 -O2 -I/path/to/tensorflow/include -I/path/to/CImg \
  src/main.cpp -L/path/to/tensorflow/lib -ltensorflow_cc -ltensorflow_framework \
  -lpthread -ldl -o fish_detector
```

Run locally with graph/labels paths (defaults are Inception example paths in the code):

```sh
./fish_detector --imagedir /path/to/images \
  --graph /path/to/tensorflow_inception_graph.pb \
  --labels /path/to/imagenet_comp_graph_label_strings.txt
```

## Coding Style & Naming Conventions

- Follow the existing style in `src/main.cpp`: 2-space indentation, braces on the same line, and short focused functions.
- Keep TensorFlow-related helper functions in PascalCase (e.g., `LoadGraph`), and variables in `snake_case` when possible.
- No formatter or linter is configured; keep changes minimal and consistent with adjacent code.

## Testing Guidelines

- There is no automated test suite in this repository.
- Validate changes by running the detector on a small image set and confirming crops appear in `/detected`.

## Commit & Pull Request Guidelines

- Commit messages in history are short, imperative, and sentence-cased (e.g., “Add build and run instructions”).
- PRs should include: a clear summary, how you tested (commands + sample data), and any output artifacts if you changed detection logic.

## Configuration Notes

- Ensure TensorFlow C++ headers/libs and `CImg.h` are installed and discoverable by your compiler.
- Default model/labels paths are hardcoded; update flags or the paths in `src/main.cpp` when using custom graphs.
