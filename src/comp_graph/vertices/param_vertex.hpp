#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <src/comp_graph/vertices/vertex.hpp>
#include <src/parameters.hpp>
#include <memory>
#include <stdexcept>

namespace fortis::comp_graph {
using fortis::parameters::Parameter;

class ParameterVertex final
    : public Vertex,
      public std::enable_shared_from_this<ParameterVertex> {
 public:
  explicit ParameterVertex(std::unique_ptr<Parameter> parameter)
      : _parameter(std::move(parameter)),
        _output_dimension(_parameter->getValue().size()) {}

  void forward() final {}
  void backward() final {
    /**
     * At the time the gradient reaches this parameter, there is no further
     * backpropagation needed. This is the actual gradient update for the
     * current parameter. You can convince yourself of this by reading through
     * this tutorial on backpropagation.
     * https://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/readings/L06%20Backpropagation.pdf
     */
    if (!_upstream_gradient.has_value()) {
      throw std::runtime_error(
          "Cannot propagate the gradient backward without "
          "setting the upstream gradient first.");
    }
    auto trainable_parameter_count = _parameter->getParameterCount();
    auto total_gradients = _upstream_gradient.value().size() *
                           _upstream_gradient.value().at(0).size();

    if (trainable_parameter_count != total_gradients) {
      throw std::runtime_error(
          "Invalid gradient encountered during parameter update. The total "
          "number of trainable parameters does not match the total number of "
          "gradient updates.");
    }
    _parameter->setGradient(_upstream_gradient.value());
  }
  inline std::vector<std::vector<float>> getOutput() const final {
    return _parameter->getValue();
  }

  std::pair<uint32_t, uint32_t> getOutputShape() const final {
    assert(!_output.empty());
    return std::make_pair(1, _output.size());
  }
  inline std::string getName() final { return "Param"; }

 private:
  std::shared_ptr<Vertex> applyOperation() { return shared_from_this(); }
  std::unique_ptr<Parameter> _parameter;
  uint32_t _output_dimension;

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Vertex>(this), _parameter);
  }
};

}  // namespace fortis::comp_graph

CEREAL_REGISTER_TYPE(fortis::comp_graph::ParameterVertex)