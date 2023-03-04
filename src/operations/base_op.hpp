#pragma once

#include <cereal/access.hpp>
#include <cereal/types/polymorphic.hpp>
#include <memory>
#include <string>
#include <vector>

namespace fortis {

class Vertex;
using VertexPointer = std::shared_ptr<Vertex>;

struct Expression {

  void operator()(VertexPointer &vertex) { _value = std::move(vertex); }
  VertexPointer _value;
};

class Vertex {
public:
  virtual ~Vertex() = default;

  virtual void forward();
  virtual void backward();

protected:
  virtual std::shared_ptr<Expression> applyOperation();

private:
  std::vector<Expression> _edges;
  friend class cereal::access;
  template <typename Archive> void serialize(Archive &archive) {
    (void)archive;
  }
};


} // namespace fortis