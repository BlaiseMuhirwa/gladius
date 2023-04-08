#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <src/comp_graph/vertices/vertex.hpp>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fortis::comp_graph {

using fortis::comp_graph::Vertex;
using fortis::comp_graph::VertexPointer;

class Summation final : public Vertex,
                        public std::enable_shared_from_this<Summation> {
 public:
  Summation(VertexPointer left_input, VertexPointer right_input)
      : _left_input(std::move(left_input)),
        _right_input(std::move(right_input)) {
    auto left_input_shape = _left_input->getOutputShape();
    auto right_input_shape = _right_input->getOutputShape();
    bool dimensions_match =
        (left_input_shape.first == right_input_shape.first) &&
        (left_input_shape.second == right_input_shape.second);

    if (!dimensions_match) {
      throw std::invalid_argument(
          "Dimension mismatch for the inputs to summation vertex. Make sure "
          "that the two inputs have the same dimensions.");
    }
    _output_length = left_input_shape.second;
    _output.reserve(_output_length);
  }

  /* Move constructor */
  Summation(Summation&& other) noexcept
      : _left_input(std::move(other._left_input)),
        _right_input(std::move(other._right_input)),
        _gradient(std::move(other._gradient)) {}

  void forward() final {
    assert(_output.empty());
    applyOperation();
  }
  /**
   * Let \phi(x, y) = x + y be the operation represented by this vertex
   * for two input vectors x and y. Then, we observe that the Jacobian
   * of \phi w.r.t either x or y is the identity matrix. Thus, the local
   * gradient is the identity so that the upstream gradient parameter
   * is passed backward to either input vertex for downstream gradients
   * computations.
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

    // Checks if this is the first time backpropagating through this vertex
    // On the first pass we populate the derivative, which I_n x gradient
    // i.e., the upstream gradient is copied over
    if (_gradient.empty()) {
      _gradient = _upstream_gradient.value();
    } else {
      for (uint32_t row_index = 0; row_index < _gradient.size(); row_index++) {
        for (uint32_t col_index = 0; col_index < _gradient.at(0).size();
             col_index++) {
          _gradient[row_index][col_index] +=
              _upstream_gradient.value()[row_index][col_index];
        }
      }
    }
    _left_input->setUpstreamGradient(/* gradient = */ _gradient);
    _right_input->setUpstreamGradient(/* gradient = */ _gradient);
  }

  inline std::string getName() final { return "Summation"; }

  std::pair<uint32_t, uint32_t> getOutputShape() const final {
    return std::make_pair(1, _output_length);
  }

 private:
  std::shared_ptr<Vertex> applyOperation() final {
    auto left_output_vector = _left_input->getOutput().at(0);
    auto right_output_vector = _right_input->getOutput().at(0);
    auto vector_size = left_output_vector.size();

    for (uint32_t index = 0; index < vector_size; index++) {
      _output.emplace_back(left_output_vector[index] +
                           right_output_vector[index]);
    }
    return shared_from_this();
  }
  VertexPointer _left_input;
  VertexPointer _right_input;
  std::vector<std::vector<float>> _gradient;
  uint32_t _output_length;

  Summation() = default;

  friend class cereal::access;

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Vertex>(this), _left_input, _right_input,
            _gradient, _output, _upstream_gradient, _output_length);
  }
};

}  // namespace fortis::comp_graph

CEREAL_REGISTER_TYPE(fortis::comp_graph::Summation)