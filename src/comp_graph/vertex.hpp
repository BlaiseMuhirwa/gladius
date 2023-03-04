#pragma once

#include "../cereal/access.hpp"
#include "../operations/base_op.hpp"
#include <memory>
#include <vector>

namespace fortis {

using fortis::operations::OperationPointer;

class Vertex;
using VertexPointer = std::shared_ptr<Vertex>;

struct Expression {

  void operator()(const VertexPointer& vertex) {
    _vertex = std::move(vertex);
  }
  VertexPointer _vertex;
};

class Vertex {
public:
  Vertex(OperationPointer &operation) : _operation(operation) {}

  void operator()(std::vector<VertexPointer>& incoming_edges) {
    _edges = std::move(incoming_edges);
  }

  void forward();
  void backward();

private:
  OperationPointer _operation;
  std::vector<std::shared_ptr<Vertex>> _edges;

  Vertex() {}
  friend class cereal::access;
  template <typename Archive> void serialize(Archive &archive) {
    archive(_operation, _edges);
  }
};


} // namespace fortis