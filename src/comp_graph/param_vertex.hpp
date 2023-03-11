#include "types/base_class.hpp"
#include <cereal/access.hpp>
#include <cereal/types/polymorphic.hpp>
#include <memory>
#include <src/comp_graph/vertex.hpp>
#include <src/parameters.hpp>

namespace fortis::comp_graph {
using fortis::parameters::Parameter;

class ParameterVertex : public Vertex,
                        public std::enable_shared_from_this<ParameterVertex> {
public:
  explicit ParameterVertex(std::shared_ptr<Parameter> parameter)
      : _parameter(std::move(parameter)) {}

  void forward() final { return; }
  void backward(const std::vector<std::vector<float>> &gradient) final {
    /**
     * At the time the gradient reaches this parameter, there is no further
     * backpropagation needed. This is the actual gradient update for the
     * current parameter. You can convince yourself of this by reading through
     * this tutorial on backpropagation.
     * https://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/readings/L06%20Backpropagation.pdf
     */
    _parameter->setGradient(gradient);
  }
  std::vector<std::vector<float>> getOutput() const final {
    return _parameter->value();
  }

private:
  std::shared_ptr<Vertex> applyOperation() { return shared_from_this(); }
  std::shared_ptr<Parameter> _parameter;

  friend class cereal::access;
  template <typename Archive> void serialize(Archive &archive) {
    archive(cereal::base_class<Vertex>(this), _parameter);
  }
};

} // namespace fortis::comp_graph

CEREAL_REGISTER_TYPE(fortis::comp_graph::ParameterVertex)