
#include "graph.hpp"
#include "param_vertex.hpp"
#include "trainer.hpp"
#include <cassert>
#include <iostream>
#include <memory>
#include <parameters.hpp>
#include <src/comp_graph/vertices/activ_functions.hpp>
#include <src/comp_graph/vertices/input_vertex.hpp>
#include <src/comp_graph/vertices/vertex.hpp>
#include <src/model.hpp>
#include <src/parameters.hpp>
#include <src/utils.hpp>
#include <tuple>

using fortis::comp_graph::InputVertex;
using fortis::comp_graph::ParameterVertex;
using fortis::comp_graph::TanHActivation;
using fortis::comp_graph::Vertex;
using fortis::parameters::Parameter;
using fortis::parameters::ParameterPointer;
using fortis::parameters::ParameterType;

static inline const char *TRAIN_DATA = "data/train-images-idx3-ubyte";
static inline const char *TRAIN_LABELS = "data/train-labels-idx1-ubyte";
static inline const float LEARNING_RATE = 0.0001f;

// Parameter &createParameter(std::vector < std::vector<uint32_t> & input,
//                            std::vector<uint32_t> &label) {
//   std::vector < std::vector<float> normalized_input;
//   std::for_each(input.begin(), input.end(),
//                 [&](const std::vector<uint32_t> &vec) {
//                   std::vector<float> normalized_vec;
//                   for (const auto &value : xvec) {
//                     float normalized_value = value / 255.f;
//                     normalized_vec.push_back(normalized_value);
//                   }
//                   normalized_input.push_back(std::move(normalized_vec));
//                 });
//   return Parameter(std::move(normalized_input));
// }

void initializeParameters(
    std::unique_ptr<fortis::Model> &model,
    const std::unordered_map<ParameterType, std::vector<uint32_t>>
        &parameters) {

  for (const auto &[param_type, dimensions] : parameters) {
    if (param_type == ParameterType::BiasParameter) {
      assert(dimensions.size() == 1);
      model->addParameter(/* dimensions = */ dimensions);
    }
  }
}

std::vector<std::tuple<ParameterType, std::vector<uint32_t>>>
defineModelParameters() {
  return {{ParameterType::WeightParameter, {256, 784}},
          {ParameterType::BiasParameter, {256}},
          {ParameterType::WeightParameter, {256, 256}},
          {ParameterType::BiasParameter, {256}},
          {ParameterType::WeightParameter, {10, 256}},
          {ParameterType::BiasParameter, {10}}};
}

int main(int argc, char **argv) {
  auto [images, labels] = fortis::readMnistDataset(
      /* image_filename = */ TRAIN_DATA, /* label_filename = */ TRAIN_LABELS);

  auto one_hot_encoded_labels = fortis::oneHotEncode(
      /* labels = */ labels, /* label_vector_dimension = */ 10);

  std::unique_ptr<fortis::Model> model = std::make_unique<fortis::Model>();
  auto weights_and_biases_parameters = defineModelParameters();

  initializeParameters(/* model = */ model, /* parameters = */
                       {{ParameterType::BiasParameter, {1000}},
                        {ParameterType::WeightParameter, {1000, 784}}});

  auto gradient_descent_trainer = fortis::trainers::GradientDescentTrainer(
      /* model = */ model, /* learning_rate = */ LEARNING_RATE);
  auto computation_graph = std::make_shared<fortis::comp_graph::Graph>();

  for (auto &image : images) {
    auto normalized_input = fortis::normalizeInput<uint32_t>(
        /* input_vector = */ image, /* normalizer = */ 255.f);

    computation_graph->renewComputationGraph();

    computation_graph->addVertex(
        std::make_shared<InputVertex>(normalized_input));

    computation_graph->addVertex(
        std::make_shared<ParameterVertex>(model->getParameterByName("")));
  }

  return 0;
}