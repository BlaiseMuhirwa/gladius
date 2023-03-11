#include <cereal/access.hpp>
#include <memory>
#include <src/comp_graph/vertex.hpp>
#include <vector>

namespace fortis::comp_graph {

using fortis::comp_graph::Vertex;

class InputVertex : public Vertex,
                    public std::enable_shared_from_this<InputVertex> {
public:
  explicit InputVertex(std::shared_ptr<std::vector<float>> input)
      : _input(std::move(input)) {}

  void forward() final { return; }
  void backward(const std::vector<std::vector<float>> &gradient) final {
    return;
  }

  std::vector<std::vector<float>> getOutput() const final { return {*_input}; }

private:
  std::shared_ptr<Vertex> applyOperation() final { return shared_from_this(); }
  std::shared_ptr<std::vector<float>> _input;

  InputVertex() {}
  friend class cereal::access;

  template <typename Archive> void serialize(Archive &archive) {
    archive(_input);
  }
};

} // namespace fortis::comp_graph