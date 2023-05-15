#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <_types/_uint32_t.h>
#include <_types/_uint8_t.h>
#include <src/comp_graph/vertices/vertex.hpp>
#include <src/utils.hpp>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <format>
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
    auto output_shape = getOutputShape();
    _local_gradient = std::vector<float>(output_shape.second, 0.F);
  }
  SoftMaxActivation(const SoftMaxActivation&) = delete;
  SoftMaxActivation& operator=(const SoftMaxActivation&) = delete;
  SoftMaxActivation& operator=(SoftMaxActivation&&) = delete;

  void forward() final {
    assert(_output.empty());
    _logits = _incoming_edges.at(0)->getOutput().at(0);
    applyOperation();
  }

  /**
   * The predicted label corresponds to the argmax over all elements in the
   * output. We return the index of the maximizer
   * TODO: Maybe parallelize this implementation?
   */
  uint32_t getPredictedLabel() const {
    assert(!_output.empty());

    // Get iterator pointing to the maximum element
    auto max_iterator = std::max_element(_output.begin(), _output.end());
    return static_cast<uint32_t>(std::distance(_output.begin(), max_iterator));
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
  void backward(std::optional<std::vector<float>>& upstream_grad) final {
    if (!upstream_grad.has_value()) {
      throw std::runtime_error(
          "Cannot propagate the gradient backward without "
          "setting the upstream gradient first.");
    }

    assert(upstream_grad.value().size() == _output.size());
    assert(!_output.empty());

    // We will assume that the softmax operation is only connected to
    // one loss function so that backpropagation through this vertex
    // only happens once. As the types of supported models increase,
    // this requirement can be relaxed.
    assert(_local_gradient.has_value());

    auto num_dimensions = _logits.size();
    std::vector<std::vector<float>> jacobian_matrix(
        num_dimensions, std::vector<float>(num_dimensions, 0.F));

    for (uint32_t row_index = 0; row_index < num_dimensions; row_index++) {
      for (uint32_t col_index = 0; col_index < num_dimensions; col_index++) {
        if (row_index == col_index) {
          jacobian_matrix[row_index][col_index] =
              _output[row_index] * (1 - _output[row_index]);
          // jacobian_matrix[row_index][col_index] =
          //     -(_output[row_index] * _output[col_index]);
        } else {
          // jacobian_matrix[row_index][col_index] =
          //     -(_output[row_index] * _output[col_index]);
        }
      }
    }

    for (uint32_t col_index = 0; col_index < num_dimensions; col_index++) {
      (*_local_gradient)[col_index] = fortis::utils::innerProduct(  // NOLINT
          /* vector = */ upstream_grad.value(),
          /* matrix = */ jacobian_matrix,
          /* col_index = */ col_index);
    }

    // std::cout << "[softmax-backward]" << std::endl;

    auto previous_vertex = _incoming_edges.at(0);
    previous_vertex->backward(/* upstream_grad = */ _local_gradient);

    // std::cout << "[softmax-finished upstream grads updates]" << std::endl;
  }

  std::pair<uint32_t, uint32_t> getOutputShape() const final {
    return _incoming_edges.at(0)->getOutputShape();
  }

  std::string getName() final { return "SoftMax"; }

 private:
  /**
   * Computes the softmax operation for the input logits. To prevent
   * overflow/underflow we use the log-sum-exponent trick to normalize the
   * exponentiated values. For more on this, check out the following article:
   * https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
   */
  std::shared_ptr<Vertex> applyOperation() final {
    // Let's use the log-softmax instead of the regular softmax
    // to penalize the model more when it predicts the incorrect
    // class
    float sum_exps = 0.F;
    // std::cout << "[logits]" << " ";
    // for (auto& logit : _logits) {
    //   std::cout << logit << " ";
    // }
    // std::cout << "\n";
    auto max_element = std::max_element(_logits.begin(), _logits.end());

    // std::cout << "[max-element] " << *max_element << std::endl;
    std::for_each(_logits.begin(), _logits.end(),
                  [&sum_exps, &max_element](float logit) {
                    sum_exps += exp(logit - *max_element);
                  });
    // std::cout << "[sum-exps] " << sum_exps << std::endl;
    // float lse = log(sum_exps);
    for (auto& logit : _logits) {
      float log_softmax = exp(logit - *max_element) / sum_exps;
      // std::cout << "[softmax] = " << log_softmax << std::endl;
      // float softmax_normalized_logit = exp(logit - *max_element) / sum_exps;
      _output.push_back(log_softmax);
    }
    return shared_from_this();
  }

  std::vector<VertexPointer> _incoming_edges;
  std::vector<float> _logits;

  SoftMaxActivation() = default;

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Vertex>(this), _output, _local_gradient,
            _incoming_edges);
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
    auto output_shape = getOutputShape();
    _local_gradient = std::vector<float>(output_shape.second, 0.F);
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
  void backward(std::optional<std::vector<float>>& upstream_grad) final {
    if (!upstream_grad.has_value()) {
      throw std::runtime_error(
          "Cannot propagate the gradient backward without "
          "setting the upstream gradient first.");
    }
    assert(!_output.empty());
    assert(_local_gradient.has_value());
    assert(upstream_grad.value().size() == _output.size());

    auto dimensions = _output.size();
    if (!_jacobian.has_value()) {
      _jacobian = std::vector<std::vector<float>>(
          dimensions, std::vector<float>(dimensions, 0.F));

      auto input_vector = _incoming_edges.at(0)->getOutput();
      for (uint32_t index = 0; index < dimensions; index++) {
        (*_jacobian)[index][index] =
            input_vector.at(0).at(index) > 0.F ? 1.0 : 0.F;
      }
    }

    // This is not quite correct since ReLU is not differentiable at 0,
    // but we will just set it to 0 at 0
    for (uint32_t col_index = 0; col_index < dimensions; col_index++) {
      (*_local_gradient)[col_index] += fortis::utils::innerProduct(
          /* vector = */ upstream_grad.value(),
          /* matrix = */ _jacobian.value(),
          /* col_index = */ col_index);
    }

    // std::cout << "[relu-backward]" << std::endl;

    auto previous_vertex = _incoming_edges.at(0);
    previous_vertex->backward(/* upstream_grad = */ _local_gradient);
  }

  inline std::string getName() final { return "ReLU"; }

  std::pair<uint32_t, uint32_t> getOutputShape() const final {
    return _incoming_edges.at(0)->getOutputShape();
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

  ReLUActivation() = default;

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Vertex>(this), _incoming_edges, _jacobian,
            _local_gradient, _output);
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
    auto output_shape = getOutputShape();
    _local_gradient = std::vector<float>(output_shape.second, 0.F);
  }

  TanHActivation(const TanHActivation&) = delete;
  TanHActivation& operator=(const TanHActivation&) = delete;
  TanHActivation& operator=(TanHActivation&&) = delete;

  void forward() final {
    assert(!_incoming_edges.empty());
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
  void backward(std::optional<std::vector<float>>& upstream_grad) final {
    if (!upstream_grad.has_value()) {
      throw std::runtime_error(
          "Cannot propagate the gradient backward without "
          "setting the upstream gradient first.");
    }
    assert(upstream_grad.value().size() == _output.size());
    assert(!_output.empty());

    auto dimensions = _output.size();
    if (!_jacobian.has_value()) {
      _jacobian = std::vector<std::vector<float>>(
          dimensions, std::vector<float>(dimensions, 0.F));

      // auto input_vector = _incoming_edges.at(0)->getOutput();
      for (uint32_t index = 0; index < dimensions; index++) {
        auto current_activation = _output[index];
        float tanh_derivative = 1 - (current_activation * current_activation);
        (*_jacobian)[index][index] = tanh_derivative;
      }
    }
    for (uint32_t col_index = 0; col_index < dimensions; col_index++) {
      (*_local_gradient)[col_index] += fortis::utils::innerProduct(  // NOLINT
          /* vector = */ upstream_grad.value(),
          /* matrix = */ _jacobian.value(),
          /* col_index = */ col_index);
    }

    auto previous_vertex = _incoming_edges.at(0);
    previous_vertex->backward(/* upstream_grad = */ _local_gradient);
  }

  inline std::string getName() final { return "TanH"; }

  std::pair<uint32_t, uint32_t> getOutputShape() const final {
    return _incoming_edges.at(0)->getOutputShape();
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

  TanHActivation() = default;

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Vertex>(this), _incoming_edges, _jacobian,
            _local_gradient, _output);
  }
};

}  // namespace fortis::comp_graph

#include <cereal/archives/binary.hpp>

CEREAL_REGISTER_TYPE(fortis::comp_graph::TanHActivation)
CEREAL_REGISTER_TYPE(fortis::comp_graph::ReLUActivation)
CEREAL_REGISTER_TYPE(fortis::comp_graph::SoftMaxActivation)
