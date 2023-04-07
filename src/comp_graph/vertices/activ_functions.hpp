#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <_types/_uint32_t.h>
#include <src/comp_graph/vertices/vertex.hpp>
#include <src/utils.hpp>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <format>
#include <math.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace fortis::comp_graph {
using fortis::comp_graph::Vertex;

class SoftMaxActivation final
    : public Vertex,
      public std::enable_shared_from_this<SoftMaxActivation> {
 public:
  explicit SoftMaxActivation(std::vector<VertexPointer>&& incoming_edges)
      : _incoming_edges(incoming_edges) {
    if (_incoming_edges.size() != 1) {
      throw std::runtime_error(
          "SoftMax activation function expects a single vector as "
          "input. Received " +
          std::to_string(incoming_edges.size()) + " vectors.");
    }
    _logits = _incoming_edges.at(0)->getOutput().at(0);
  }
  SoftMaxActivation(const SoftMaxActivation&) = delete;
  SoftMaxActivation& operator=(const SoftMaxActivation&) = delete;
  SoftMaxActivation& operator=(SoftMaxActivation&&) = delete;

  void forward() final {
    assert(!_logits.empty());
    assert(_output.empty());

    applyOperation();
  }

  /**
   * The predicted label corresponds to the argmax over all elements in the
   * output. We return the index of the maximizer
   */
  float getPredictedLabel() const {
    // Get iterator pointing to the maximum element
    auto max_iterator = std::max_element(_output.begin(), _output.end());
    return (float)std::distance(_output.begin(), max_iterator);
  }

  /**
   * This implementation computes the partial derivatives of the loss function
   * w.r.t any logit.
   * Suppose the logits vector has size k. Then, the gradient (jacobian) of the
   * softmax function w.r.t any logit is a (k x k) matrix where [Dsof(x)]_ij is
   * given by
   *        [Dsof(x)]_ij = sof(x)_i(1 - sof(x)_j) if i = j
   *                     = - sof(x)_i sof(x)_j    if i \neq j
   *
   * For the dimensions to match, this implies that the input gradient from the
   * loss MUST have dimensions (1 x k), 1 since the loss is a scalar value. We
   * then use the chain rule to compute the partials of the loss function w.r.t
   * the logits
   */
  void backward() final {
    if (!_upstream_gradient.has_value()) {
      throw std::runtime_error(
          "Cannot propagate the gradient backward without "
          "setting the upstream gradient first.");
    }
    assert(_upstream_gradient.value().size() == 1);
    assert(_upstream_gradient.value().at(0).size() == _output.size());
    assert(!_output.empty());
    // We will assume that the softmax operation is only connected to
    // one loss function so that backpropagation through this vertex
    // only happens once. As the types of supported models increase,
    // this requirement can be relaxed.
    assert(_local_gradient.empty());

    auto num_dimensions = _logits.size();
    std::vector<std::vector<float>> jacobian_matrix(
        num_dimensions, std::vector<float>(num_dimensions, 0.f));

    for (uint32_t row_index = 0; row_index < num_dimensions; row_index++) {
      for (uint32_t col_index = 0; col_index < num_dimensions; col_index++) {
        if (row_index == col_index) {
          jacobian_matrix[row_index][col_index] =
              -(_output[row_index] * _output[col_index]);
        } else {
          jacobian_matrix[row_index][col_index] =
              _output[row_index] * (1 - _output[row_index]);
        }
      }
    }

    _local_gradient = std::vector<std::vector<float>>(
        1, std::vector<float>(num_dimensions, 0.f));

    for (uint32_t col_index = 0; col_index < num_dimensions; col_index++) {
      _local_gradient[0][col_index] = fortis::utils::dotProduct(
          /* vector = */ _upstream_gradient.value().at(0),
          /* matrix = */ jacobian_matrix,
          /* col_index = */ col_index);
    }
    auto previous_vertex = _incoming_edges.at(0);
    previous_vertex->setUpstreamGradient(/* gradient = */ _local_gradient);
  }

  std::pair<uint32_t, uint32_t> getOutputShape() const final {
    assert(!_output.empty());
    return std::make_pair(1, _output.size());
  }

  std::string getName() final { return "SoftMax"; }

 private:
  /**
   * Computes the softmax operation for the input logits
   */
  std::shared_ptr<Vertex> applyOperation() final {
    auto size = _logits.size();

    float sum_exponents = 0.f;
    std::for_each(_logits.begin(), _logits.end(),
                  [&](float value) { sum_exponents += exp(value); });

    for (auto& logit : _logits) {
      float softmax_normalized_logit = exp(logit) / sum_exponents;
      _output.push_back(softmax_normalized_logit);
    }
    return shared_from_this();
  }

  std::vector<VertexPointer> _incoming_edges;
  std::vector<float> _logits;

  SoftMaxActivation() {}

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Vertex>(this), _output, _local_gradient,
            _upstream_gradient, _incoming_edges);
  }
};

class ReLUActivation final
    : public Vertex,
      public std::enable_shared_from_this<ReLUActivation> {
 public:
  explicit ReLUActivation(std::vector<VertexPointer>&& incoming_edges)
      : _incoming_edges(incoming_edges), _jacobian(std::nullopt) {
    if (_incoming_edges.size() != 1) {
      throw std::runtime_error(
          "ReLU activation function expects a single vector as "
          "input. Received " +
          std::to_string(_incoming_edges.size()) + " vectors.");
    }
  }

  ReLUActivation(const ReLUActivation&) = delete;
  ReLUActivation& operator=(const ReLUActivation&) = delete;
  ReLUActivation& operator=(ReLUActivation&&) = delete;

  void forward() final {
    assert(!_incoming_edges.empty());
    assert(_output.empty());
    applyOperation();
  }

  /**
   * Note that the Jacobian matrix is a diagonal matrix that is
   * almost identical to the Identity matrix.
   * TODO: Computing the Jacobian matrix is completely unnecessary.
   *       Remove it.
   */
  void backward() final {
    if (!_upstream_gradient.has_value()) {
      throw std::runtime_error(
          "Cannot propagate the gradient backward without "
          "setting the upstream gradient first.");
    }
    assert(!_output.empty());
    assert(_upstream_gradient.value().size() == 1);
    assert(_upstream_gradient.value().at(0).size() == _output.size());

    auto dimensions = _output.size();
    if (!_jacobian.has_value()) {
      _jacobian = std::vector<std::vector<float>>(
          dimensions, std::vector<float>(dimensions, 0.f));

      auto input_vector = _incoming_edges.at(0)->getOutput();
      for (uint32_t index = 0; index < dimensions; index++) {
        (*_jacobian)[index][index] =
            input_vector.at(0).at(index) > 0.f ? 1.0 : 0.f;
      }
    }
    // This is not quite correct since ReLU is not differentiable at 0,
    // but we will just set it to 0 at 0
    for (uint32_t col_index = 0; col_index < dimensions; col_index++) {
      _local_gradient[0][col_index] += fortis::utils::dotProduct(
          /* vector = */ _upstream_gradient.value().at(0),
          /* matrix = */ _jacobian.value(),
          /* col_index = */ col_index);
    }
    auto previous_vertex = _incoming_edges.at(0);
    previous_vertex->setUpstreamGradient(/* gradient = */ _local_gradient);
  }

  inline std::string getName() final { return "ReLU"; }

  std::pair<uint32_t, uint32_t> getOutputShape() const final {
    assert(!_output.empty());
    return std::make_pair(1, _output.size());
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
  std::optional<std::vector<std::vector<float>>> _jacobian;

  ReLUActivation() {}

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Vertex>(this), _incoming_edges, _jacobian,
            _local_gradient, _upstream_gradient, _output);
  }
};

class TanHActivation final
    : public Vertex,
      public std::enable_shared_from_this<TanHActivation> {
  /**
   * The TanH activation computes the hyperbolic tangent function
   * Recall that for any input value x, TanH(x) is given by
   *
   *          TanH(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}
   **/
 public:
  explicit TanHActivation(std::vector<VertexPointer>&& incoming_edges)
      : _incoming_edges(incoming_edges), _jacobian(std::nullopt) {
    if (_incoming_edges.size() != 1) {
      throw std::runtime_error(
          "TanH activation function expects a single vector as "
          "input. Received " +
          std::to_string(_incoming_edges.size()) + " vectors.");
    }
  }

  TanHActivation(const TanHActivation&) = delete;
  TanHActivation& operator=(const TanHActivation&) = delete;
  TanHActivation& operator=(TanHActivation&&) = delete;

  void forward() final {
    assert(_incoming_edges.size() != 0);
    assert(_output.empty());
    applyOperation();
  }

  /**
   * The local gradient for TanH is expressed as follows:
   * - numerator: 4 * exp(-2x)
   * - denominator: (1+exp(-2x))^2
   * simplifying, we get that d(tanh(x))/dx = 1 - [tanh(x)^2]
   *
   */
  void backward() final {
    if (!_upstream_gradient.has_value()) {
      throw std::runtime_error(
          "Cannot propagate the gradient backward without "
          "setting the upstream gradient first.");
    }
    assert(_upstream_gradient.value().size() == 1);
    assert(_upstream_gradient.value().at(0).size() == _output.size());
    assert(!_output.empty());

    auto dimensions = _output.size();
    if (!_jacobian.has_value()) {
      _jacobian = std::vector<std::vector<float>>(
          dimensions, std::vector<float>(dimensions, 0.f));

      auto input_vector = _incoming_edges.at(0)->getOutput();
      for (uint32_t index = 0; index < dimensions; index++) {
        auto current_activation = _output[index];
        float tanh_derivative = 1 - (current_activation * current_activation);
        (*_jacobian)[index][index] = tanh_derivative;
      }
    }
    for (uint32_t col_index = 0; col_index < dimensions; col_index++) {
      _local_gradient[0][col_index] += fortis::utils::dotProduct(
          /* vector = */ _upstream_gradient.value().at(0),
          /* matrix = */ _jacobian.value(),
          /* col_index = */ col_index);
    }

    auto previous_vertex = _incoming_edges.at(0);
    previous_vertex->setUpstreamGradient(/* gradient = */ _local_gradient);
  }

  inline std::string getName() final { return "TanH"; }

  std::pair<uint32_t, uint32_t> getOutputShape() const final {
    assert(!_output.empty());
    return std::make_pair(1, _output.size());
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
  std::optional<std::vector<std::vector<float>>> _jacobian;

  TanHActivation() {}

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Vertex>(this), _incoming_edges, _jacobian,
            _local_gradient, _upstream_gradient, _output);
  }
};

}  // namespace fortis::comp_graph

#include <cereal/archives/binary.hpp>

CEREAL_REGISTER_TYPE(fortis::comp_graph::TanHActivation)
CEREAL_REGISTER_TYPE(fortis::comp_graph::ReLUActivation)
CEREAL_REGISTER_TYPE(fortis::comp_graph::SoftMaxActivation)
