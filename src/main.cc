
#include <cassert>
#include <memory>
#include <parameters.hpp>
#include <src/comp_graph/vertices/activ_functions.hpp>
#include <src/comp_graph/vertices/vertex.hpp>
#include <src/parameters.hpp>

using fortis::comp_graph::TanHActivation;
using fortis::comp_graph::Vertex;
using fortis::parameters::Parameter;
using fortis::parameters::ParameterPointer;

static inline const char *TRAIN_DATA = "data/train-images-idx3-ubyte";
static inline const char *TRAIN_LABELS = "data/train-labels-idx1-ubyte";

uint32_t swapEndian(uint32_t val) {
  val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
  return (val << 16) | (val >> 16);
}

void readMNIST(const char *image_filename, const char *label_filename) {
  std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
  std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);

  // Read the magic and the meta data
  uint32_t magic;
  uint32_t num_items;
  uint32_t num_labels;
  uint32_t rows;
  uint32_t cols;

  image_file.read(reinterpret_cast<char *>(&magic), 4);
  magic = swapEndian(magic);
  if (magic != 2051) {
    std::cout << "Incorrect image file magic: " << magic << std::endl;
    return;
  }

  label_file.read(reinterpret_cast<char *>(&magic), 4);
  magic = swapEndian(magic);
  if (magic != 2049) {
    std::cout << "Incorrect image file magic: " << magic << std::endl;
    return;
  }
  image_file.read(reinterpret_cast<char *>(&num_items), 4);
  num_items = swapEndian(num_items);
  label_file.read(reinterpret_cast<char *>(&num_labels), 4);
  num_labels = swapEndian(num_labels);
  if (num_items != num_labels) {
    std::cout << "image file nums should equal to label num" << std::endl;
    return;
  }

  image_file.read(reinterpret_cast<char *>(&rows), 4);
  rows = swapEndian(rows);
  image_file.read(reinterpret_cast<char *>(&cols), 4);
  cols = swapEndian(cols);

  std::cout << "image and label num is: " << num_items << std::endl;
  std::cout << "image rows: " << rows << ", cols: " << cols << std::endl;

  char label;
  char *pixels = new char[rows * cols];

  for (int item_id = 0; item_id < num_items; ++item_id) {
    // read image pixel
    image_file.read(pixels, rows * cols);
    // read label
    label_file.read(&label, 1);

    std::string sLabel = std::to_string(int(label));
    std::cout << "label is: " << sLabel << std::endl;
    // // convert it to cv Mat, and show it
    // cv::Mat image_tmp(rows, cols, CV_8UC1, pixels);
    // // resize bigger for showing
    // cv::resize(image_tmp, image_tmp, cv::Size(100, 100));
    // cv::imshow(sLabel, image_tmp);
    // cv::waitKey(0);
  }

  delete[] pixels;
}

int main(int argc, char **argv) {
  readMNIST(TRAIN_DATA, TRAIN_LABELS);
  return 0;
}