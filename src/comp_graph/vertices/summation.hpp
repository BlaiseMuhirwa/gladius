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
   * is passed backward to both input vertices for downstream gradient
   * computations.
   */
  void backward(std::optional<std::vector<float>>& upstream_grad) final {
    if (!upstream_grad.has_value()) {
      throw std::runtime_error(
          "Cannot propagate the gradient backward without "
          "setting the upstream gradient first.");
    }
    assert(!_output.empty());
    assert(upstream_grad.value().size() == _output.size());

    // std::cout << "[summation-vertex-backward]" << std::endl;

    _left_input->backward(/* upstream_grad = */ upstream_grad);
    _right_input->backward(/* upstream_grad = */ upstream_grad);
  }

  inline std::string getName() final { return "Summation"; }

  std::pair<uint32_t, uint32_t> getOutputShape() const final {
    return std::make_pair(1, _output_length);
  }

 private:
  std::shared_ptr<Vertex> applyOperation() final {
    auto left_output_vector = _left_input->getOutput();
    auto right_output_vector = _right_input->getOutput();
    auto vector_size = left_output_vector.size();

    for (uint32_t index = 0; index < vector_size; index++) {
      auto sum = left_output_vector[index] + right_output_vector[index];
      // std::cout << "[summation val] " << sum << std::endl;
      _output.push_back(sum);
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
            _gradient, _output, _output_length);
  }
};

}  // namespace fortis::comp_graph

CEREAL_REGISTER_TYPE(fortis::comp_graph::Summation)