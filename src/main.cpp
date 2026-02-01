#include <fstream>

#include <vector>

#include <dirent.h>

#include "tensorflow/cc/ops/const_op.h"

#include "tensorflow/cc/ops/image_ops.h"

#include "tensorflow/cc/ops/standard_ops.h"

#include "tensorflow/core/framework/graph.pb.h"

#include "tensorflow/core/framework/tensor.h"

#include "tensorflow/core/graph/default_device.h"

#include "tensorflow/core/graph/graph_def_builder.h"

#include "tensorflow/core/lib/core/errors.h"

#include "tensorflow/core/lib/core/stringpiece.h"

#include "tensorflow/core/lib/core/threadpool.h"

#include "tensorflow/core/lib/io/path.h"

#include "tensorflow/core/lib/strings/stringprintf.h"

#include "tensorflow/core/platform/init_main.h"

#include "tensorflow/core/platform/logging.h"

#include "tensorflow/core/platform/types.h"

#include "tensorflow/core/public/session.h"

#include "tensorflow/core/util/command_line_flags.h"

#define cimg_display 0

#define cimg_use_jpeg 1

#include "CImg.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::int32;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
Status ReadLabelsFile(string file_name, std::vector<string> *result,
                      size_t *found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    return tensorflow::errors::NotFound("Labels file ", file_name,
                                        " not found.");
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return Status::OK();
}

Status ReadTensorFromImageObject(cimg_library::CImg<float> &input_image,
                                 const float input_mean, const float input_std,
                                 std::vector<Tensor> *out_tensors) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops; // NOLINT(build/namespaces)

  string input_name = "file_reader";
  string output_name = "normalized";

  int input_height = 299; // используем размер изображения базы Inception
  int input_width = 299; //
  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 3;
  Output image_reader;

  cimg_library::CImg<float> resized_image =
      input_image.get_resize(input_width, input_height);
  resized_image = (resized_image - input_mean) / input_std;
  tensorflow::Tensor inputImg(
      tensorflow::DT_FLOAT,
      tensorflow::TensorShape({1, input_height, input_width, wanted_channels}));
  auto inputImageMapped = inputImg.tensor<float, 4>();

  // Copy all the data over
  for (int y = 0; y < input_height; ++y) {
    for (int x = 0; x < input_width; ++x) {
      inputImageMapped(0, y, x, 0) = *resized_image.data(x, y, 0, 0);
      inputImageMapped(0, y, x, 1) = *resized_image.data(x, y, 0, 1);
      inputImageMapped(0, y, x, 2) = *resized_image.data(x, y, 0, 2);
    }
  }

  out_tensors->push_back(inputImg);
  /*
  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
                                               tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({}, {inputImageMapped}, {}, out_tensors));*/
  return Status::OK();
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(string graph_file_name,
                 std::unique_ptr<tensorflow::Session> *session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

// Analyzes the output of the Inception graph to retrieve the highest scores and
// their positions in the tensor, which correspond to categories.
Status GetTopLabels(const std::vector<Tensor> &outputs, int how_many_labels,
                    Tensor *indices, Tensor *scores) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops; // NOLINT(build/namespaces)

  string output_name = "top_k";
  TopKV2(root.WithOpName(output_name), outputs[0], how_many_labels);
  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensors.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));

  // The TopK node returns two outputs, the scores and their original indices,
  // so we have to append :0 and :1 to specify them both.
  std::vector<Tensor> out_tensors;
  TF_RETURN_IF_ERROR(session->Run({}, {output_name + ":0", output_name + ":1"},
                                  {}, &out_tensors));
  *scores = out_tensors[0];
  *indices = out_tensors[1];
  return Status::OK();
}

// Given the output of a model run, and the name of a file containing the labels
// this prints out the top five highest-scoring values.
Status PrintTopLabels(const std::vector<Tensor> &outputs,
                      string labels_file_name) {
  std::vector<string> labels;
  size_t label_count;
  Status read_labels_status =
      ReadLabelsFile(labels_file_name, &labels, &label_count);
  if (!read_labels_status.ok()) {
    LOG(ERROR) << read_labels_status;
    return read_labels_status;
  }
  const int how_many_labels = std::min(5, static_cast<int>(label_count));
  Tensor indices;
  Tensor scores;
  TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
  tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
  for (int pos = 0; pos < how_many_labels; ++pos) {
    const int label_index = indices_flat(pos);
    const float score = scores_flat(pos);
    LOG(INFO) << labels[label_index] << " (" << label_index << "): " << score;
  }
  return Status::OK();
}

bool IsFish(const std::vector<Tensor> &outputs, string labels_file_name) {
  bool is_fish = false;
  std::vector<string> labels;
  size_t label_count;
  Status read_labels_status =
      ReadLabelsFile(labels_file_name, &labels, &label_count);
  const int how_many_labels = std::min(5, static_cast<int>(label_count));
  Tensor indices;
  Tensor scores;
  GetTopLabels(outputs, how_many_labels, &indices, &scores);
  tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();

  float combined_score = 0.0;
  for (int pos = 0; pos < how_many_labels; ++pos) {
    const int label_index = indices_flat(pos);

    // check top label id's for something fishy

    const float score = scores_flat(pos);
    // if ((label_index>=442)&&(label_index<=458)) combined_score += score;
    if (score > 0.95)
      combined_score += score;
    LOG(INFO) << labels[label_index] << " (" << label_index << "): " << score;
  }

  if (combined_score > 0.95)
    is_fish = true;

  return is_fish;
}

// This is a testing function that returns whether the top label index is the
// one that's expected.
Status CheckTopLabel(const std::vector<Tensor> &outputs, int expected,
                     bool *is_expected) {
  *is_expected = false;
  Tensor indices;
  Tensor scores;
  const int how_many_labels = 1;
  TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
  if (indices_flat(0) != expected) {
    LOG(ERROR) << "Expected label #" << expected << " but got #"
               << indices_flat(0);
    *is_expected = false;
  } else {
    *is_expected = true;
  }
  return Status::OK();
}

std::string NameMaker(std::string a, int x0, int y0, int x1, int y1) {
  std::string ret;
  std::size_t pos = a.find(".");

  ret = a.substr(0, pos);
  ret += "_";
  ret += std::to_string(x0);
  ret += "_";
  ret += std::to_string(y0);
  ret += "_";
  ret += std::to_string(x1);
  ret += "_";
  ret += std::to_string(y1);
  ret += ".jpg";

  return ret;
}

// we will test subimages for image pattern recognition here
Status TestImageAreas(cimg_library::CImg<float> &raw_image,
                      std::unique_ptr<tensorflow::Session> &session,
                      string &image_file_name, string &image_dir,
                      const float input_mean, const float input_std,
                      string labels, string input_layer, string output_layer) {
  bool is_fish;
  int width, height;
  int x0, y0, x1, y1, k = 0;

  width = raw_image.width();
  height = raw_image.height();

  std::string detected_image_dir =
      tensorflow::io::JoinPath(image_dir, "detected");

  for (y0 = 0; y0 < height; y0 = y0 + height / 4) {
    for (x0 = 0; x0 < width; x0 = x0 + height / 4) {
      // координаты другой крайней точки присваиваем как сумму первоначальной
      // координаты и половины длины/ширины картинки
      x1 = x0 + height / 2;
      y1 = y0 + height / 2;
      k++;
      // проверяем, не выходят ли крайние точки за пределы картинки. если
      // выходят, то обрезаем ее по границе

      if (x1 > width) {
        x1 = width;
      }
      if (y1 > height) {
        y1 = height;
      }
      // выводим получившиеся координаты
      LOG(INFO) << "Крайние координаты картинки " << k << " равны:\n";
      LOG(INFO) << "(" << x0 << ";" << y0 << ")\n";
      LOG(INFO) << "(" << x1 << ";" << y1 << ")\n";
      std::vector<Tensor> resized_tensors;

      cimg_library::CImg<float> cropped_image =
          raw_image.get_crop(x0, y0, x1, y1);

      Status read_tensor_status = ReadTensorFromImageObject(
          cropped_image, input_mean, input_std, &resized_tensors);
      if (!read_tensor_status.ok()) {
        LOG(ERROR) << read_tensor_status;
        return read_tensor_status;
      }
      const Tensor &resized_tensor = resized_tensors[0];

      // Actually run the image through the model.
      std::vector<Tensor> outputs;
      Status run_status = session->Run({{input_layer, resized_tensor}},
                                       {output_layer}, {}, &outputs);
      if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        return run_status;
      }

      // Do something interesting with the results we've generated.
      bool print_status = IsFish(outputs, labels);
      if (print_status == true) {
        LOG(INFO) << "FISH IS DETECTED!!!";
        // check is fish here
        string croppedFileName = NameMaker(image_file_name, x0, y0, x1, y1);

        string image_path =
            tensorflow::io::JoinPath(detected_image_dir, croppedFileName);
        cropped_image.save_jpeg(image_path.c_str());
      }
    }
  }
  return Status::OK();
}

int main(int argc, char *argv[]) {
  string image = "";
  string image_dir = "";
  string graph = "tensorflow/examples/label_image/data/"
                 "tensorflow_inception_graph.pb";
  string labels = "tensorflow/examples/label_image/data/"
                  "imagenet_comp_graph_label_strings.txt";
  int32 input_width = 299;
  int32 input_height = 299;
  int32 input_mean = 128;
  int32 input_std = 128;
  string input_layer = "Mul";
  string output_layer = "softmax";
  bool self_test = false;
  string root_dir = "";
  std::vector<Flag> flag_list = {
      Flag("image", &image, "image to be processed"),
      Flag("imagedir", &image_dir, "image directory to be processed"),
      Flag("graph", &graph, "graph to be executed"),
      Flag("labels", &labels, "name of file containing labels"),
      Flag("input_width", &input_width, "resize image to this width in pixels"),
      Flag("input_height", &input_height,
           "resize image to this height in pixels"),
      Flag("input_mean", &input_mean, "scale pixel values to this mean"),
      Flag("input_std", &input_std, "scale pixel values to this std deviation"),
      Flag("input_layer", &input_layer, "name of input layer"),
      Flag("output_layer", &output_layer, "name of output layer"),
      Flag("self_test", &self_test, "run a self test"),
      Flag("root_dir", &root_dir,
           "interpret image and graph file names relative to this directory"),
  };
  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  // First we load and initialize the model.
  std::unique_ptr<tensorflow::Session> session;
  string graph_path = tensorflow::io::JoinPath(root_dir, graph);
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }

  // do some cicle for files in dir
  DIR *dir = opendir(image_dir.c_str());
  if (dir != NULL) {
    struct dirent *ent;
    int k = 0;
    while ((ent = readdir(dir)) != NULL) {
      // Get the image from disk as a float array of numbers, resized and
      // normalized to the specifications the main graph expects.
      string image_file_name = ent->d_name;

      if (ent->d_type != DT_REG ||
          (image_file_name.find(".jpg") == string::npos &&
           image_file_name.find(".jpeg") == string::npos &&
           image_file_name.find(".png") == string::npos))
        continue;

      std::vector<Tensor> resized_tensors;
      string image_path = tensorflow::io::JoinPath(image_dir, image_file_name);

      LOG(INFO) << "image file: " << image_path << "\n";

      cimg_library::CImg<float> raw_image;
      raw_image.load_jpeg(image_path.c_str());
      int raw_image_width = raw_image.width();
      int raw_image_hight = raw_image.height();

      // разрежем картинку на части

      TestImageAreas(raw_image, session, image_file_name, image_dir, input_mean,
                     input_std, labels, input_layer, output_layer);
      k++;

      LOG(INFO) << "image file width: " << raw_image_width << "\n";
      LOG(INFO) << "image file height: " << raw_image_hight << "\n";

      /*Status read_tensor_status =
      ReadTensorFromImageFile(image_path, input_height, input_width, input_mean,
                              input_std, &resized_tensors);
      if (!read_tensor_status.ok()) {
          LOG(ERROR) << read_tensor_status;
          return -1;
      }
      const Tensor& resized_tensor = resized_tensors[0];

      // Actually run the image through the model.
      std::vector<Tensor> outputs;
      Status run_status = session->Run({{input_layer, resized_tensor}},
                                       {output_layer}, {}, &outputs);
      if (!run_status.ok()) {
          LOG(ERROR) << "Running model failed: " << run_status;
          return -1;
      }
      */

      // Do something interesting with the results we've generated.
      /*Status print_status = PrintTopLabels(outputs, labels);
      if (!print_status.ok()) {
          LOG(ERROR) << "Running print failed: " << print_status;
          return -1;
      }*/
    }
    closedir(dir);
  } else {
    LOG(ERROR) << "Error opening directory\n";
    return -1;
  }

  // end cicle

  return 0;
}