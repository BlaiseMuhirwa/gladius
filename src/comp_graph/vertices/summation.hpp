#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <memory>
#include <src/comp_graph/vertices/vertex.hpp>
#include <stdexcept>
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

    if (_left_input->getOutputSize() != _right_input->getOutputSize()) {
      throw std::invalid_argument(
          "Dimension mismatch for the inputs to summation vertex. Make sure "
          "that the two inputs have the same dimensions.");
    }
    _output.reserve(_left_input->getOutputSize());
  }

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
  void backward(const std::optional<std::vector<std::vector<float>>> &gradient =
                    std::nullopt) final {
    assert(gradient.has_value());

    // Checks if this is the first time backpropagating through this vertex
    // On the first pass we populate the derivative, which I_n x gradient
    // i.e., the upstream gradient is copied over
    if (_gradient.empty()) {
      // _gradient = std::vector<float>(size_to_allocate, 1.0);
      _gradient = gradient.value();
    } else {
      assert(_gradient.size() == gradient.value().size());
      assert(_gradient.at(0).size() == gradient.value().at(0).size());

      for (uint32_t row_index = 0; row_index < _gradient.size(); row_index++) {
        for (uint32_t col_index = 0; col_index < _gradient.at(0).size();
             col_index++) {
          _gradient[row_index][col_index] +=
              gradient.value()[row_index][col_index];
        }
      }
    }
  }

  inline std::string getName() final { return "Summation"; }

  constexpr uint32_t getOutputSize() const final {
    return _left_input->getOutputSize();
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

  Summation() {}
  friend class cereal::access;

  template <typename Archive> void serialize(Archive &archive) {
    archive(cereal::base_class<Vertex>(this), _left_input, _right_input,
            _output, _gradient);
  }
};

} // namespace fortis::comp_graph

CEREAL_REGISTER_TYPE(fortis::comp_graph::Summation)