#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <src/comp_graph/vertices/vertex.hpp>
#include <src/params/parameters.hpp>
#include <memory>
#include <stdexcept>

namespace fortis::comp_graph {
using fortis::parameters::Parameter;

class ParameterVertex final
    : public Vertex,
      public std::enable_shared_from_this<ParameterVertex> {
 public:
  explicit ParameterVertex(std::shared_ptr<Parameter> parameter)
      : _parameter(std::move(parameter)) {}

  void forward() final {}
  void backward() final {
    /**
     * At the time the gradient reaches this parameter, there is no further
     * backpropagation needed. This is the actual gradient update for the
     * current parameter. You can convince yourself of this by reading through
     * this tutorial on backpropagation.
     * https://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/readings/L06%20Backpropagation.pdf
     */
    if (!_upstream_gradient.has_value() || _upstream_gradient.value().empty()) {
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
    _parameter->updateGradient(_upstream_gradient.value());
  }
  inline std::vector<std::vector<float>> getOutput() const final {
    return _parameter->getValue();
  }

  std::pair<uint32_t, uint32_t> getOutputShape() const final {
    return _parameter->getParameterShape();
  }
  inline std::string getName() final { return "Parameter"; }

 private:
  std::shared_ptr<Vertex> applyOperation() final { return shared_from_this(); }
  std::shared_ptr<Parameter> _parameter;

  ParameterVertex() = default;

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Vertex>(this), _parameter);
  }
};

}  // namespace fortis::comp_graph

CEREAL_REGISTER_TYPE(fortis::comp_graph::ParameterVertex)