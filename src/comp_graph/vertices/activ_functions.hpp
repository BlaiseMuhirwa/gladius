#pragma once

#include "vertex.hpp"
#include <cassert>
#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cstddef>
#include <format>
#include <math.h>
#include <memory>
#include <optional>
#include <stdexcept>

namespace fortis::comp_graph {
using fortis::comp_graph::Vertex;

class SoftMaxActivation final
    : public Vertex,
      public std::enable_shared_from_this<SoftMaxActivation> {
public:
  SoftMaxActivation() = default;

  std::shared_ptr<SoftMaxActivation>
  operator()(std::vector<VertexPointer> &&incoming_edges) {
    if (incoming_edges.size() != 1) {
      throw std::runtime_error(
          "ReLU activation function expects a single vector as "
          "input. Received " +
          std::to_string(incoming_edges.size()) + " vectors.");
    }
    _incoming_edges = std::move(incoming_edges);
    return shared_from_this();
  }
  void forward() final { return; }
  void backward(const std::optional<std::vector<std::vector<float>>> &gradient =
                    std::nullopt) final {
    assert(gradient.has_value());
    assert(!_output.empty());
    // We will assume that the softmax operation is only connected to
    // one loss function so that backpropagation through this vertex
    // only happens once. As the types of supported models increase,
    // this requirement can be relaxed.
    assert(_gradient.empty());

    auto vector_size = getOutputSize();
    _gradient = std::vector<float>(vector_size);

    for (uint32_t neuron_index = 0; neuron_index < vector_size;
         neuron_index++) {
      // Recall that the derivative of the softmax w.r.t any logit x_i is given
      // by derivative = softmax(x_i)(1 - softmax(x_i))
      auto softmax = _output[neuron_index];
      float derivative = softmax * (1 - softmax);
      _gradient[neuron_index] =
          gradient.value().at(0).at(neuron_index) * derivative;
    }
  }

  constexpr uint32_t getOutputSize() const final {
    assert(!_output.empty());
    return _output.size();
  }

  std::string getName() final { return "SoftMax"; }

private:
  std::shared_ptr<Vertex> applyOperation() final {
    auto incoming_edge = _incoming_edges.at(0);
    auto input_vector = incoming_edge->getOutput().at(0);
    auto size = input_vector.size();

    float sum_exponents = 0.f;
    std::for_each(input_vector.begin(), input_vector.end(),
                  [&](float value) { sum_exponents += exp(value); });

    for (auto &logit : input_vector) {
      float softmax_normalized_logit = exp(logit) / sum_exponents;
      _output.push_back(softmax_normalized_logit);
    }
    return shared_from_this();
  }

  std::vector<VertexPointer> _incoming_edges;

  friend class cereal::access;
  template <typename Archive> void serialize(Archive &archive) {
    archive(_output, _gradient, _incoming_edges);
  }
};

class ReLUActivation final
    : public Vertex,
      public std::enable_shared_from_this<ReLUActivation> {
public:
  ReLUActivation() = default;
  ReLUActivation &operator=(const ReLUActivation &) = delete;
  ReLUActivation &operator=(ReLUActivation &&) = delete;

  /**
   * We return a shared pointer to the underlying object because typically
   * this provideds us a clean interface for subsequent calls after object
   * construction. For instance, we might need to launch a forward pass
   * after calling function call operator.
   */
  std::shared_ptr<ReLUActivation>
  operator()(std::vector<VertexPointer> &&incoming_edges) {
    if (_incoming_edges.size() != 1) {
      throw std::runtime_error(
          "ReLU activation function expects a single vector as "
          "input. Received " +
          std::to_string(_incoming_edges.size()) + " vectors.");
    }
    _incoming_edges = std::move(incoming_edges);
    return shared_from_this();
  }

  void forward() final {
    assert(!_incoming_edges.empty());
    assert(_output.empty());
    applyOperation();
  }

  void backward(
      const std::optional<std::vector<std::vector<float>>> &gradient) final {
    assert(gradient.has_value());
    assert(!_output.empty());

    auto vector_size = _output.size();

    if (_gradient.empty()) {
      _gradient = std::vector<float>(vector_size);
    }

    for (uint32_t neuron_index = 0; neuron_index < vector_size;
         neuron_index++) {
      auto current_activation = _output[neuron_index];
      if (!current_activation) {
        throw std::runtime_error(
            "RelU activation is not differentiable at 0.0");
      }
      float relu_derivative = (current_activation > 0) ? 1 : 0;
      float total_derivative =
          gradient.value().at(0).at(neuron_index) * relu_derivative;
      _gradient[neuron_index] += total_derivative;
    }
  }

  inline std::string getName() final { return "ReLU"; }

  constexpr uint32_t getOutputSize() const final {
    assert(!_output.empty());
    return _output.size();
  }

private:
  std::shared_ptr<Vertex> applyOperation() final {
    auto incoming_edge = _incoming_edges.at(0);
    auto input_vector = incoming_edge->getOutput().at(0);
    auto size = input_vector.size();

    for (uint32_t neuron_index = 0; neuron_index < size; neuron_index++) {
      float relu_activation =
          input_vector[neuron_index] > 0 ? input_vector[neuron_index] : 0;
      _output.push_back(relu_activation);
    }
    return shared_from_this();
  }

  std::vector<VertexPointer> _incoming_edges;

  friend class cereal::access;
  template <typename Archive> void serialize(Archive &archive) {
    archive(cereal::base_class<Vertex>(this), _incoming_edges, _output);
  }
};

class TanHActivation final
    : public Vertex,
      public std::enable_shared_from_this<TanHActivation> {
  /**
   * The TanH activation computes the hyperbolic tangent function
   * We recall that for any input value x, TanH(x) is given by
   *
   *          TanH(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}
   **/
public:
  TanHActivation() = default;

  TanHActivation &operator=(const TanHActivation &) = delete;
  TanHActivation &operator=(TanHActivation &&) = delete;

  /**
   * We return a shared pointer to the underlying object because typically
   * this provideds us a clean interface for subsequent calls after object
   * construction. For instance, we might need to launch a forward pass
   * after calling function call operator.
   */
  void operator()(const std::vector<VertexPointer> incoming_edges) {
    if (_incoming_edges.size() != 1) {
      throw std::runtime_error(
          "ReLU activation function expects a single vector as "
          "input. Received " +
          std::to_string(_incoming_edges.size()) + " vectors.");
    }
    _incoming_edges = std::move(incoming_edges);
  }

  void forward() final {
    assert(_incoming_edges.size() != 0);
    assert(_output.empty());
    applyOperation();
  }

  /**
   * The local gradient for TanH is expressed as follows:
   * - numerator: 4 * exp(-2x)
   * - denomiator: (1+exp(-2x))^2
   * simplifying, we get that d(tanh(x))/dx = 1 - [tanh(x)^2]
   *
   */
  void backward(
      const std::optional<std::vector<std::vector<float>>> &gradient) final {
    assert(gradient.has_value());
    assert(!_output.empty());

    auto vector_size = getOutputSize();

    if (_gradient.empty()) {
      _gradient = std::vector<float>(vector_size);
    }

    for (uint32_t neuron_index = 0; neuron_index < vector_size;
         neuron_index++) {
      auto current_activation = _output[neuron_index];
      float tanh_derivative = 1 - (current_activation * current_activation);
      float total_derivative =
          gradient.value().at(0).at(neuron_index) * tanh_derivative;
      _gradient[neuron_index] += total_derivative;
    }
  }

  inline std::string getName() final { return "TanH"; }

  constexpr uint32_t getOutputSize() const final {
    assert(!_output.empty());
    return _output.size();
  }

private:
  std::shared_ptr<Vertex> applyOperation() final {

    auto incoming_edge = _incoming_edges.at(0);
    auto input_vector = incoming_edge->getOutput().at(0);
    auto size = input_vector.size();

    for (uint32_t neuron_index = 0; neuron_index < size; neuron_index++) {
      float exponential_term = exp(-2 * input_vector[neuron_index]);
      float tanh_activation = (1 - exponential_term) / (1 + exponential_term);
      _output.push_back(tanh_activation);
    }
    return shared_from_this();
  }

  std::vector<VertexPointer> _incoming_edges;

  friend class cereal::access;
  template <typename Archive> void serialize(Archive &archive) {
    archive(cereal::base_class<Vertex>(this), _incoming_edges, _output);
  }
};

} // namespace fortis::comp_graph

CEREAL_REGISTER_TYPE(fortis::comp_graph::TanHActivation)
CEREAL_REGISTER_TYPE(fortis::comp_graph::ReLUActivation)
