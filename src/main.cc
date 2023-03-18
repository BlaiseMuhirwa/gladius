
#include <cassert>
#include <iostream>
#include <memory>
#include <src/comp_graph/vertices/activ_functions.hpp>
#include <src/comp_graph/vertices/vertex.hpp>
#include <src/parameters.hpp>
#include <src/utils.hpp>


using fortis::comp_graph::TanHActivation;
using fortis::comp_graph::Vertex;
using fortis::parameters::Parameter;
using fortis::parameters::ParameterPointer;

static inline const char *TRAIN_DATA = "data/train-images-idx3-ubyte";
static inline const char *TRAIN_LABELS = "data/train-labels-idx1-ubyte";


Parameter& createParameter(std::vector<std::vector<uint32_t>& input, std::vector<uint32_t>& label) {
  std::vector<std::vector<float> normalized_input;
  std::for_each(input.begin(), input.end(), [&](const std::vector<uint32_t>& vec) {
    std::vector<float> normalized_vec;
    for(const auto& value: vec) {
      float normalized_value = value / 255.f;
      normalized_vec.push_back(normalized_value);
    }
    normalized_input.push_back(std::move(normalized_vec));
  });
  return Parameter(std::move(normalized_input));
}


int main(int argc, char **argv) {
  auto [images, labels] = fortis::readMnistDataset(TRAIN_DATA, TRAIN_LABELS);

  Parameter& createParameter(images, )

  return 0;
}