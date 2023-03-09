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

  Expression() {}
  friend class cereal::access;

  template <typename Archive> void serialize(Archive &archive) {
    archive(_value);
  }
};

class Vertex {
public:
  virtual ~Vertex() = default;

  virtual void forward();
  virtual void backward();
  virtual std::vector<std::vector<float>> getOuput();

protected:
  virtual std::shared_ptr<Vertex> applyOperation();

private:
  std::vector<std::vector<float>> _output;
  friend class cereal::access;
  template <typename Archive> void serialize(Archive &archive) {
    (void)archive;
  }
};

} // namespace fortis