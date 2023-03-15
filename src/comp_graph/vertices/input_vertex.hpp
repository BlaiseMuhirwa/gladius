#include <cereal/access.hpp>
#include <cereal/types/polymorphic.hpp>
#include <memory>
#include <src/comp_graph/vertex.hpp>
#include <vector>

namespace fortis::comp_graph {

using fortis::comp_graph::Vertex;

class InputVertex final : public Vertex,
                          public std::enable_shared_from_this<InputVertex> {
public:
  explicit InputVertex(std::shared_ptr<std::vector<float>> input)
      : _output(std::move(input)) {}

  void forward() final { return; }
  void backward(const std::optional<std::vector<std::vector<float>>> &gradient =
                    std::nullopt) final {
    return;
  }

  constexpr uint32_t getOutputDimension() const final {
    return _output->size();
  }

  std::string getName() final { return "Input"; }

  std::vector<std::vector<float>> getOutput() const override {
    return {*_output};
  }

private:
  std::shared_ptr<Vertex> applyOperation() final { return shared_from_this(); }
  std::shared_ptr<std::vector<float>> _output;

  InputVertex() {}
  friend class cereal::access;

  template <typename Archive> void serialize(Archive &archive) {
    archive(cereal::base_class<Vertex>(this), _output);
  }
};

} // namespace fortis::comp_graph

CEREAL_REGISTER_TYPE(fortis::comp_graph::InputVertex)