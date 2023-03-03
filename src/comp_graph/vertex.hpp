#pragma once

#include "../cereal/access.hpp"
#include "../operations/base_op.hpp"
#include <memory>
#include <vector>

namespace fortis {

using fortis::operations::Operation;
using fortis::operations::OperationPointer;
class Vertex {

  explicit Vertex(OperationPointer &operation) : _operation(operation) {}

  void forward();
  void backward();

private:
  OperationPointer _operation;
  std::vector<float> _edges;

  Vertex() {}
  friend class cereal::access;
  template <typename Archive> void serialize(Archive &archive) {
    archive(_operation, _edges);
  }
};

using VertexPointer = std::shared_ptr<Vertex>;

} // namespace fortis