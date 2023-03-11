#pragma once

#include <cereal/access.hpp>
#include <cereal/types/polymorphic.hpp>
#include <memory>
#include <string>
#include <vector>

namespace fortis::comp_graph {

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
  virtual std::shared_ptr<Vertex>
  setIncomingEdges(std::vector<VertexPointer> &edges) = 0;

  virtual ~Vertex() = default;

  /**
   * The forward function computes the function specified by
   * the type of the vertex in the computation graph.
   * This computation is responsible for propagating the
   * input forward in the DAG.
   */
  virtual void forward() = 0;

  /**
   * The backward function is responsible for computing local
   * gradients at a specific vertex and updating the gradient
   * of the loss function with respect to the given parameter
   * via the chain rule.
   */
  virtual void backward(const std::vector<std::vector<float>> &gradient) = 0;
  virtual std::vector<std::vector<float>> getOuput() const = 0;

protected:
  virtual std::shared_ptr<Vertex> applyOperation() = 0;

private:
  friend class cereal::access;
  template <typename Archive> void serialize(Archive &archive) {
    (void)archive;
  }
};

} // namespace fortis::comp_graph