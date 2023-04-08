
#include <_types/_uint32_t.h>
#include <gtest/gtest.h>
#include <src/comp_graph/graph.hpp>
#include <src/comp_graph/vertices/activ_functions.hpp>
#include <src/comp_graph/vertices/inner_product.hpp>
#include <src/comp_graph/vertices/input_vertex.hpp>
#include <src/comp_graph/vertices/loss.hpp>
#include <src/comp_graph/vertices/param_vertex.hpp>
#include <src/comp_graph/vertices/summation.hpp>
#include <src/comp_graph/vertices/vertex.hpp>
#include <src/model.hpp>
#include <src/parameters.hpp>
#include <src/trainer.hpp>
#include <src/utils.hpp>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <tuple>

namespace fortis::tests {

using fortis::comp_graph::CrossEntropyLoss;
using fortis::comp_graph::InnerProduct;
using fortis::comp_graph::InputVertex;
using fortis::comp_graph::ParameterVertex;
using fortis::comp_graph::ReLUActivation;
using fortis::comp_graph::SoftMaxActivation;
using fortis::comp_graph::Summation;
using fortis::comp_graph::Vertex;
using fortis::comp_graph::VertexPointer;
using fortis::parameters::ParameterType;

static inline const char* TRAIN_DATA = "data/train-images-idx3-ubyte";
static inline const char* TRAIN_LABELS = "data/train-labels-idx1-ubyte";
static inline const uint32_t NUM_LAYERS = 3;
static inline const float LEARNING_RATE = 0.0001;
static inline const float ACCURACY_THRESHOLD = 0.9;

void initializeParameters(
    std::shared_ptr<fortis::Model>& model,
    std::vector<std::tuple<ParameterType, std::vector<uint32_t>>>& parameters) {
  for (const auto& [param_type, dimensions] : parameters) {
    if (param_type == ParameterType::BiasParameter) {
      model->addParameter(/* dimension = */ dimensions[0]);
    } else {
      model->addParameter(/* dimensions = */ dimensions);
    }
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

float computeAccuracy(std::vector<float>& predicted_labels,
                      std::vector<std::vector<float>>& true_labels) {
  uint32_t correct_predictions = 0;
  for (uint32_t label_index = 0; label_index < true_labels.size();
       label_index++) {
    // We can use std::max_element since the input is guaranteed to be one-hot
    // encoded
    auto max_iterator = std::max_element(true_labels[label_index].begin(),
                                         true_labels[label_index].end());
    if (*max_iterator == predicted_labels[label_index]) {
      correct_predictions += 1;
    }
  }
  return (correct_predictions * 1.0) / predicted_labels.size();
}

TEST(FortisMLPMnist, TestAccuracyScore) {
  auto [images, labels] = fortis::utils::readMnistDataset(
      /* image_filename = */ TRAIN_DATA, /* label_filename = */ TRAIN_LABELS);

  std::vector<std::vector<float>> one_hot_encoded_labels =
      fortis::utils::oneHotEncode(
          /* labels = */ labels, /* label_vector_dimension = */ 10);

  std::shared_ptr<fortis::Model> model(new Model());
  auto weights_and_biases_parameters = defineModelParameters();

  initializeParameters(/* model = */ model,
                       /* parameters = */ weights_and_biases_parameters);

  auto gradient_descent_trainer = fortis::trainers::GradientDescentTrainer(
      /* model = */ model, /* learning_rate = */ LEARNING_RATE);

  auto computation_graph = std::make_unique<fortis::comp_graph::Graph>();

  std::vector<float> losses, predicted_labels;

  std::cout << "[STARTING TRAINING]" << std::endl;

  for (uint32_t training_sample_index = 0;
       training_sample_index < images.size(); training_sample_index++) {
    auto normalized_input = fortis::utils::normalizeInput<uint32_t>(
        /* input_vector = */ images[training_sample_index],
        /* normalizer = */ 255.F);
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
      auto inner_prod_op = std::make_shared<InnerProduct>(
          /* left_input = */ weight_parameter,
          /* right_input = */ current_activations);
      computation_graph->addVertex(inner_prod_op);

      auto summation_op = std::make_shared<Summation>(
          /* left_input = */ inner_prod_op,
          /* right_input = */ bias_parameter);
      computation_graph->addVertex(summation_op);

      if (layer_index < 2) {
        std::cout << "[relu]" << std::endl;
        std::shared_ptr<ReLUActivation> relu_activation(
            new ReLUActivation(/* incoming_edges = */ {summation_op}));
        computation_graph->addVertex(relu_activation);

        current_activations = relu_activation;
      } else {
        std::cout << "[softmax]" << std::endl;
        std::shared_ptr<SoftMaxActivation> softmax(
            new SoftMaxActivation(/* incoming_edges = */ {summation_op}));
        computation_graph->addVertex(softmax);

        std::cout << "[cross-entropy]" << std::endl;

        auto loss_function = std::make_shared<CrossEntropyLoss>(
            /* input_vertex = */ softmax, /* label = */ label);
        computation_graph->addVertex(loss_function);
      }
      std::cout << "[layer done]" << std::endl;
    }

    auto [predicted_label, loss] = computation_graph->launchForwardPass();
    losses.push_back(loss);
    predicted_labels.push_back(predicted_label);
    computation_graph->launchBackwardPass();
    gradient_descent_trainer.takeDescentStep();
  }

  std::cout << "[END TRAINING]" << std::endl;

  auto accuracy = computeAccuracy(predicted_labels, one_hot_encoded_labels);

  ASSERT_GE(accuracy, ACCURACY_THRESHOLD);
}

}  // namespace fortis::tests