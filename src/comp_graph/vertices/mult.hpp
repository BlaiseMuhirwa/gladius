#pragma once

#include <_types/_uint32_t.h>
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

class Multiplier final : public Vertex,
                         public std::enable_shared_from_this<Multiplier> {
public:
  Multiplier(VertexPointer left_input, VertexPointer right_input)
      : _left_input(std::move(left_input)),
        _right_input(std::move(right_input)) {

    if (_left_input->getOutputDimension() !=
        _right_input->getOutputDimension()) {
      throw std::invalid_argument(
          "Dimension mismatch for the inputs to Multiplier vertex. Make sure "
          "that the two inputs have the same dimensions.");
    }
    _output.reserve(_left_input->getOutputDimension());
  }

  void forward() final {
    assert(_output.empty());
    applyOperation();
  }
  void backward(const std::optional<std::vector<std::vector<float>>> &gradient =
                    std::nullopt) final {
    assert(gradient.has_value());

    // Checks if this is the first time backpropagating through this vertex
    // On the first pass we populate the derivative, which 1.0
    if (_gradient.empty()) {
      auto size_to_allocate = _output.size();
      _gradient = std::vector<float>(size_to_allocate, 1.0);
    }

    for (uint32_t neuron_index = 0;
         neuron_index < gradient.value().at(0).size(); neuron_index++) {
      _gradient[neuron_index] += gradient.value().at(0).at(neuron_index);
    }
  }

  inline std::string getName() final { return "Input"; }

  constexpr uint32_t getOutputDimension() const final {
    return _left_input->getOutputDimension();
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

  Multiplier() {}
  friend class cereal::access;

  template <typename Archive> void serialize(Archive &archive) {
    archive(cereal::base_class<Vertex>(this), _left_input, _right_input,
            _output, _gradient);
  }
};

} // namespace fortis::comp_graph

CEREAL_REGISTER_TYPE(fortis::comp_graph::Multiplier)