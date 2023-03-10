#include <cereal/access.hpp>
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
  void backward() final { return; }
  std::vector<std::vector<float>> getOutput() const final {
    return _parameter->value();
  }

private:
  std::shared_ptr<Vertex> applyOperation() { return shared_from_this(); }
  std::shared_ptr<Parameter> _parameter;

  friend class cereal::access;
  template <typename Archive> void serialize(Archive &archive) {
    archive(_parameter);
  }
};

} // namespace fortis::comp_graph