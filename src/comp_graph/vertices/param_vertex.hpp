#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <memory>
#include <src/comp_graph/vertices/vertex.hpp>
#include <src/parameters.hpp>

namespace fortis::comp_graph {
using fortis::parameters::Parameter;

class ParameterVertex final
    : public Vertex,
      public std::enable_shared_from_this<ParameterVertex> {
public:
  explicit ParameterVertex(std::shared_ptr<Parameter> parameter)
      : _parameter(std::move(parameter)),
        _output_dimension(_parameter->getValue().size()) {}

  void forward() final { return; }
  void backward(
      const std::optional<std::vector<std::vector<float>>> &gradient) final {
    /**
     * At the time the gradient reaches this parameter, there is no further
     * backpropagation needed. This is the actual gradient update for the
     * current parameter. You can convince yourself of this by reading through
     * this tutorial on backpropagation.
     * https://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/readings/L06%20Backpropagation.pdf
     */
    assert(gradient.has_value());
    _parameter->setGradient(gradient.value());
  }
  inline std::vector<std::vector<float>> getOutput() const final {
    return _parameter->getValue();
  }

  constexpr uint32_t getOutputDimension() const final {
    return _output_dimension;
  }

  inline std::string getName() final { return "Param"; }

private:
  std::shared_ptr<Vertex> applyOperation() { return shared_from_this(); }
  std::shared_ptr<Parameter> _parameter;
  uint32_t _output_dimension;

  friend class cereal::access;
  template <typename Archive> void serialize(Archive &archive) {
    archive(cereal::base_class<Vertex>(this), _parameter);
  }
};

} // namespace fortis::comp_graph

CEREAL_REGISTER_TYPE(fortis::comp_graph::ParameterVertex)