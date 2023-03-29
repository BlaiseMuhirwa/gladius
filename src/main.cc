
#include <_types/_uint32_t.h>
#include <cassert>
#include <iostream>
#include <memory>
#include <parameters.hpp>
#include <src/comp_graph/graph.hpp>
#include <src/comp_graph/vertices/activ_functions.hpp>
#include <src/comp_graph/vertices/input_vertex.hpp>
#include <src/comp_graph/vertices/loss.hpp>
#include <src/comp_graph/vertices/mult.hpp>
#include <src/comp_graph/vertices/param_vertex.hpp>
#include <src/comp_graph/vertices/summation.hpp>
#include <src/comp_graph/vertices/vertex.hpp>
#include <src/model.hpp>
#include <src/parameters.hpp>
#include <src/trainer.hpp>
#include <src/utils.hpp>
#include <tuple>

using fortis::comp_graph::CrossEntropyLoss;
using fortis::comp_graph::InputVertex;
using fortis::comp_graph::Multiplier;
using fortis::comp_graph::ParameterVertex;
using fortis::comp_graph::ReLUActivation;
using fortis::comp_graph::Summation;
using fortis::comp_graph::Vertex;
using fortis::comp_graph::VertexPointer;
using fortis::parameters::ParameterType;

static inline const char *TRAIN_DATA = "data/train-images-idx3-ubyte";
static inline const char *TRAIN_LABELS = "data/train-labels-idx1-ubyte";
static inline const uint32_t NUM_LAYERS = 3;
static inline const float LEARNING_RATE = 0.0001f;

void initializeParameters(
    std::unique_ptr<fortis::Model> &model,
    std::vector<std::tuple<ParameterType, std::vector<uint32_t>>> &parameters) {

  for (const auto &[param_type, dimensions] : parameters) {
    if (param_type == ParameterType::BiasParameter) {
      assert(dimensions.size() == 1);
    } else {
      assert(dimensions.size() == 2);
    }
    model->addParameter(/* dimensions = */ dimensions);
  }
}

/**
 * Parameters needed to build a 3-layer Feed-forward model
 * for MNIST classification.
 * The following configuration represents a model with a total of
 * 269,322 trainable parameters.
 */
std::vector<std::tuple<ParameterType, std::vector<uint32_t>>>
defineModelParameters() {
  return {{ParameterType::WeightParameter, {256, 784}},
          {ParameterType::BiasParameter, {256}},
          {ParameterType::WeightParameter, {256, 256}},
          {ParameterType::BiasParameter, {256}},
          {ParameterType::WeightParameter, {10, 256}},
          {ParameterType::BiasParameter, {10}}};
}

float computeAccuracy(std::vector<std::vector<float>> &probabilities,
                      std::vector<std::vector<uint32_t>> &labels);

int main(int argc, char **argv) {
  auto [images, labels] = fortis::readMnistDataset(
      /* image_filename = */ TRAIN_DATA, /* label_filename = */ TRAIN_LABELS);

  std::vector<std::vector<uint32_t>> one_hot_encoded_labels =
      fortis::oneHotEncode(
          /* labels = */ labels, /* label_vector_dimension = */ 10);

  std::unique_ptr<fortis::Model> model = std::make_unique<fortis::Model>();
  auto weights_and_biases_parameters = defineModelParameters();

  initializeParameters(/* model = */ model,
                       /* parameters = */ weights_and_biases_parameters);

  auto gradient_descent_trainer = fortis::trainers::GradientDescentTrainer(
      /* model = */ model, /* learning_rate = */ LEARNING_RATE);
  auto computation_graph = std::make_unique<fortis::comp_graph::Graph>();

  for (uint32_t training_sample_index = 0;
       training_sample_index < images.size(); training_sample_index++) {
    auto normalized_input = fortis::normalizeInput<uint32_t>(
        /* input_vector = */ images[training_sample_index],
        /* normalizer = */ 255.f);
    auto label = one_hot_encoded_labels[training_sample_index];

    computation_graph->clearComputationGraph();

    // Creates input vertex and adds it to the graph
    auto input_vertex = std::make_shared<InputVertex>(normalized_input);
    computation_graph->addVertex(input_vertex);

    std::shared_ptr<Vertex> current_activations = input_vertex;

    for (uint32_t layer_index = 0; layer_index < NUM_LAYERS; layer_index++) {
      uint32_t weight_index = (layer_index * 2);
      uint32_t bias_index = weight_index + 1;

      // Creates W_i parameter vertex and adds it to the graph
      auto weight_parameter = std::make_shared<ParameterVertex>(
          model->getParameterByID(/* param_id = */ weight_index));
      computation_graph->addVertex(weight_parameter);

      // Creates b_i parameter vertex and adds it to the graph
      auto bias_parameter = std::make_shared<ParameterVertex>(
          model->getParameterByID(/* param_id = */ bias_index));
      computation_graph->addVertex(bias_parameter);

      // Representation of the forward-prop through the ith-layer
      auto multiplication_op = std::make_shared<Multiplier>(
          /* left_input = */ weight_parameter,
          /* right_input = */ current_activations);
      computation_graph->addVertex(multiplication_op);

      auto summation_op = std::make_shared<Summation>(
          /* left_input = */ multiplication_op,
          /* right_input = */ bias_parameter);
      computation_graph->addVertex(summation_op);

      if (layer_index < 2) {
        ReLUActivation relu_op;
        auto relu_activations = relu_op(/* incoming_edges = */ {summation_op});
        computation_graph->addVertex(relu_activations);

        current_activations = relu_activations;
      } else {
        auto loss_function = std::make_shared<CrossEntropyLoss>(
            /* input_vertex = */ summation_op, /* label = */ label);
      }
    }

    auto loss = computation_graph->launchForwardPass();
    computation_graph->launchBackwardPass();
    gradient_descent_trainer.takeDescentStep();
  }

  return 0;
}
