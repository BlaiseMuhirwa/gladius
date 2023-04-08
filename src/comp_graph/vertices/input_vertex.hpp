#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <src/comp_graph/vertices/vertex.hpp>
#include <memory>
#include <vector>

namespace fortis::comp_graph {

using fortis::comp_graph::Vertex;

class InputVertex final : public Vertex,
                          public std::enable_shared_from_this<InputVertex> {
 public:
  explicit InputVertex(std::vector<float>& input)
      : _output(std::make_shared<std::vector<float>>(std::move(input))) {}

  InputVertex(const InputVertex&) = delete;
  InputVertex& operator=(const InputVertex&) = delete;

  void forward() override {}
  void backward() override {}

  inline std::string getName() final { return "Input"; }

  inline std::vector<std::vector<float>> getOutput() const override {
    return {*_output};
  }

  std::pair<uint32_t, uint32_t> getOutputShape() const final {
    assert(!_output->empty());
    return std::make_pair(1, _output->size());
  }

 private:
  std::shared_ptr<Vertex> applyOperation() final { return shared_from_this(); }
  std::shared_ptr<std::vector<float>> _output;

  InputVertex() = default;
  friend class cereal::access;

  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Vertex>(this), _output);
  }
};

}  // namespace fortis::comp_graph

CEREAL_REGISTER_TYPE(fortis::comp_graph::InputVertex)