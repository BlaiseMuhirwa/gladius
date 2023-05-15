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
#include <src/params/parameters.hpp>
#include <src/trainers/trainer.hpp>
#include <src/utils.hpp>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fortis::tests {

using fortis::comp_graph::CrossEntropyLoss;
using fortis::comp_graph::InnerProduct;
using fortis::comp_graph::InputVertex;
using fortis::comp_graph::ParameterVertex;
using fortis::comp_graph::ReLUActivation;
// using fortis::comp_graph::SoftMaxActivation;
using fortis::comp_graph::Summation;
// using fortis::comp_graph::TanHActivation;
using fortis::comp_graph::Vertex;
using fortis::comp_graph::VertexPointer;
using fortis::parameters::ParameterType;
using fortis::trainers::GradientDescentTrainer;

static inline const char* TRAIN_DATA = "data/train-images-idx3-ubyte";
static inline const char* TRAIN_LABELS = "data/train-labels-idx1-ubyte";
static inline constexpr uint32_t NUM_LAYERS = 1;
static inline constexpr float LEARNING_RATE = 0.001;
static inline constexpr float ACCURACY_THRESHOLD = 0.9;

// Total training examples: 60000
static inline constexpr uint32_t FETCH_COUNT = 300;
static inline constexpr uint32_t TRAIN_COUNT = 0.75 * FETCH_COUNT;

static void initializeParameters(
    std::shared_ptr<fortis::Model>& model,
    std::vector<std::tuple<ParameterType, std::vector<uint32_t>>>& parameters) {
  for (auto& [param_type, dimensions] : parameters) {
    model->addParameter(/* dimensions = */ std::move(dimensions));
  }
}

/**
 * Parameters needed to build a 3-layer Feed-forward model
 * for MNIST classification.
 * The following configuration represents a model with a total of
 * 269,322 trainable parameters.
 */
static std::vector<std::tuple<ParameterType, std::vector<uint32_t>>>
defineModelParameters() {
  return {{ParameterType::WeightParameter, {10, 784}},
          {ParameterType::BiasParameter, {10}},
          // {ParameterType::WeightParameter, {256, 256}},
          // {ParameterType::BiasParameter, {256}},
          // {ParameterType::WeightParameter, {10, 128}},
          // {ParameterType::BiasParameter, {10}}
          };
}

static float computeAccuracy(std::vector<uint32_t>& predicted_labels,
                             std::vector<std::vector<uint32_t>>& true_labels) {
  assert(predicted_labels.size() == true_labels.size());
  uint32_t total_labels = true_labels.size();
  uint32_t correct_predictions = 0;

  for (uint32_t label_index = 0; label_index < total_labels; label_index++) {
    // We can use std::max_element since the input is guaranteed to be one-hot
    // encoded
    auto max_iterator = std::max_element(true_labels[label_index].begin(),
                                         true_labels[label_index].end());
    if (*max_iterator == predicted_labels[label_index]) {
      correct_predictions += 1;
    }
  }
  return static_cast<float>(correct_predictions) / total_labels;
}

static std::unique_ptr<fortis::comp_graph::Graph> buildComputationGraph(
    std::shared_ptr<fortis::Model>& model, std::vector<float>& input_sample,
    std::vector<uint32_t>& label) {
  auto computation_graph = std::make_unique<fortis::comp_graph::Graph>();
  auto input_vertex = std::make_shared<InputVertex>(input_sample);
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

    if (layer_index < NUM_LAYERS - 1) {
      // std::cout << "[relu]" << std::endl;
      std::shared_ptr<ReLUActivation> tanh_activation(
          new ReLUActivation(/* incoming_edges = */ {summation_op}));
      computation_graph->addVertex(tanh_activation);

      current_activations = tanh_activation;
    } else {
      // std::cout << "[softmax]" << std::endl;
      // std::shared_ptr<SoftMaxActivation> softmax(
      //     new SoftMaxActivation(/* incoming_edges = */ {summation_op}));
      // computation_graph->addVertex(softmax);

      auto loss_function = std::make_shared<CrossEntropyLoss>(
          /* input_vertex = */ summation_op, /* label = */ label);
      computation_graph->addVertex(loss_function);
    }
  }

  return computation_graph;
}

static void train(
    const std::shared_ptr<GradientDescentTrainer>& trainer,
    std::vector<std::pair<std::vector<float>, std::vector<uint32_t>>>& dataset,
    uint32_t epochs = 5) {
  auto model = trainer->getModel();
  float total_loss = 0.F;

  for (uint32_t epoch = 0; epoch < epochs; epoch++) {
#pragma omp parallel for default(none) \
    shared(dataset, model, total_loss, trainer) 
    for (auto [input, label] : dataset) {
      trainer->zeroOutGradients();
      auto graph = buildComputationGraph(model, input, label);
      auto [predicted_label, loss] = graph->launchForwardPass();

      total_loss += loss;

      auto num_vertices = graph->getVerticesCount();
      auto loss_vertex =
          graph->getVertexAtIndex(/* vertex_id = */ num_vertices - 1);

      std::optional<std::vector<float>> grad = std::nullopt;
      loss_vertex->backward(grad);

      // #pragma omp critical
      trainer->takeDescentStep();
    
    }

    auto avg_loss = total_loss / dataset.size();
    std::cout << "[epoch-" << epoch + 1 << "-loss] = " << avg_loss << std::endl;

    total_loss = 0.F;
  }  
}  

static float evaluate(
    std::shared_ptr<fortis::Model>&& model,
    std::vector<std::pair<std::vector<float>, std::vector<uint32_t>>>&
        dataset) {
  std::vector<uint32_t> predictions;
  std::vector<std::vector<uint32_t>> labels;

  for (auto& [input, label] : dataset) {
    labels.push_back(std::move(label));

    auto graph = buildComputationGraph(model, input, label);
    auto [predicted_label, loss] = graph->launchForwardPass();

    predictions.push_back(predicted_label);
  }

  auto accuracy = computeAccuracy(predictions, labels);

  return accuracy;
}

TEST(FortisMLPMnist, TestAccuracyScore) {
  auto dataset = fortis::utils::readMNISTDataset(
      /* images_filename = */ TRAIN_DATA, /* labels_filename = */ TRAIN_LABELS,
      /* chunk_size = */ FETCH_COUNT);

  auto rng = std::default_random_engine{};
  std::shuffle(std::begin(dataset), std::end(dataset), rng);

  auto training_data =
      std::vector<std::pair<std::vector<float>, std::vector<uint32_t>>>(
          dataset.begin(), dataset.begin() + TRAIN_COUNT);
  auto testing_data =
      std::vector<std::pair<std::vector<float>, std::vector<uint32_t>>>(
          dataset.begin() + TRAIN_COUNT + 1, dataset.end());

  std::shared_ptr<fortis::Model> model(new Model());
  auto weights_and_biases_parameters = defineModelParameters();

  initializeParameters(/* model = */ model,
                       /* parameters = */ weights_and_biases_parameters);

  auto gradient_descent_trainer = std::make_shared<GradientDescentTrainer>(
      /* model = */ model, /* learning_rate = */ LEARNING_RATE);

  train(gradient_descent_trainer, training_data, 200);

  auto accuracy = evaluate(gradient_descent_trainer->getModel(),
  testing_data);

  ASSERT_GE(accuracy, ACCURACY_THRESHOLD);

}  

}  // namespace fortis::tests