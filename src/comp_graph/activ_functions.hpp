
#include "vertex.hpp"
#include <cassert>
#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cstddef>
#include <format>
#include <math.h>
#include <memory>
#include <stdexcept>

namespace fortis::comp_graph {
using fortis::comp_graph::Expression;
using fortis::comp_graph::Vertex;

class SigmoidActivation;

class ReLUActivation : public Vertex,
                       public std::enable_shared_from_this<ReLUActivation> {
public:
  ReLUActivation() {}
  ReLUActivation &operator=(const TanHActivation &) = delete;
  ReLUActivation &operator=(TanHActivation &&) = delete;
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
    assert(!_incoming_edges.empty());
    assert(_forward_output.empty());
    applyOperation();
  }

  void backward(const std::vector<float> &gradient) final {
    assert(!gradient.empty());
    assert(!_forward_output.empty());
    assert(_gradient.empty());

    auto vector_size = _forward_output.size();

    for (uint32_t neuron_index = 0; neuron_index < vector_size;
         neuron_index++) {
      auto current_activation = _forward_output[neuron_index];
      if (!current_activation) {
        throw std::runtime_error(
            "RelU activation is not differentiable at 0.0");
      }
      float relu_derivative = (current_activation > 0) ? 1 : 0;
      float total_derivative = gradient[neuron_index] * relu_derivative;
      _gradient.push_back(total_derivative);
    }
  }

  std::vector<std::vector<float>> getOuput() const final {
    return {_forward_output};
  }

private:
  std::shared_ptr<Vertex> applyOperation() final {
    auto incoming_edge = _incoming_edges.at(0);
    auto input_vector = incoming_edge->getOuput().at(0);
    auto size = input_vector.size();

    for (uint32_t neuron_index = 0; neuron_index < size; neuron_index++) {
      float relu_activation =
          input_vector[neuron_index] > 0 ? input_vector[neuron_index] : 0;
      _forward_output.push_back(relu_activation);
    }
    return shared_from_this();
  }

  std::vector<VertexPointer> _incoming_edges;
  std::vector<float> _forward_output;
  std::vector<float> _gradient;

  friend class cereal::access;
  template <typename Archive> void serialize(Archive &archive) {
    archive(cereal::base_class<Vertex>(this), _incoming_edges, _forward_output);
  }
};

class TanHActivation : public Vertex,
                       public std::enable_shared_from_this<TanHActivation> {
  /**
   * The TanH activation computes the hyperbolic tangent function
   * We recall that for any input value x, TanH(x) is given by
   *
   *          TanH(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}
   **/
public:
  TanHActivation() {}

  TanHActivation &operator=(const TanHActivation &) = delete;
  TanHActivation &operator=(TanHActivation &&) = delete;
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
    assert(_forward_output.empty());
    applyOperation();
  }

  /**
   * The local gradient for TanH is expressed as follows:
   * - numerator: 4 * exp(-2x)
   * - denomiator: (1+exp(-2x))^2
   * simplifying, we get that d(tanh(x))/dx = 1 - [tanh(x)^2]
   *
   */
  void backward(const std::vector<float> &gradient) final {
    assert(!gradient.empty());
    assert(!_forward_output.empty());
    assert(_gradient.empty());

    auto vector_size = _forward_output.size();

    for (uint32_t neuron_index = 0; neuron_index < vector_size;
         neuron_index++) {
      auto current_activation = _forward_output[neuron_index];
      float tanh_derivative = 1 - (current_activation * current_activation);
      float total_derivative = gradient[neuron_index] * tanh_derivative;
      _gradient.push_back(total_derivative);
    }
  }

  std::vector<std::vector<float>> getOuput() const final {
    return {_forward_output};
  }

private:
  std::shared_ptr<Vertex> applyOperation() final {

    auto incoming_edge = _incoming_edges.at(0);
    auto input_vector = incoming_edge->getOuput().at(0);
    auto size = input_vector.size();

    for (uint32_t neuron_index = 0; neuron_index < size; neuron_index++) {
      float exponential_term = exp(-2 * input_vector[neuron_index]);
      float tanh_activation = (1 - exponential_term) / (1 + exponential_term);
      _forward_output.push_back(tanh_activation);
    }
    return shared_from_this();
  }

  std::vector<VertexPointer> _incoming_edges;
  std::vector<float> _forward_output;
  std::vector<float> _gradient;

  friend class cereal::access;
  template <typename Archive> void serialize(Archive &archive) {
    archive(cereal::base_class<Vertex>(this), _incoming_edges, _forward_output);
  }
};

} // namespace fortis::comp_graph

CEREAL_REGISTER_TYPE(fortis::comp_graph::TanHActivation)
CEREAL_REGISTER_TYPE(fortis::comp_graph::ReLUActivation)
CEREAL_REGISTER_TYPE(fortis::comp_graph::SigmoidActivation)
