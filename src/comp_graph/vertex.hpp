#pragma once

#include <_types/_uint32_t.h>
#include <cereal/access.hpp>
#include <cereal/types/polymorphic.hpp>
#include <memory>
#include <optional>
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
  virtual ~Vertex() = default;

  /**
   * The forward function computes the function specified by
   * the type of the vertex in the computation graph.
   * This computation is responsible for propagating the
   * input forward in the DAG. In addition, the forward method
   * will also optionally allocate the size for the gradient vector
   * since we want to use std::vector::operator[], which may be invoked
   * multiple times during the backward pass. This will happen
   * when a vertex has at least two child vertices.
   */
  virtual void forward() = 0;

  /**
   * The backward function is responsible for computing local
   * gradients at a specific vertex and updating the gradient
   * of the loss function with respect to the given parameter
   * via the chain rule.
   */
  virtual void backward(const std::optional<std::vector<std::vector<float>>>
                            &gradient = std::nullopt) = 0;

  /**
   * Returns the output of the forward pass through the vertex.
   */
  virtual std::vector<std::vector<float>> getOutput() const {
    assert(!_output.empty());
    return {_output};
  }

  /**
   * Returns the name of the corresponding operation implemented by
   * the vertex.
   */
  virtual std::string getName() = 0;

  /**
   * Returns the gradient of the loss function with respect to
   * the operation computed by the vertex.
   */
  virtual std::vector<std::vector<float>> getGradient() const {
    assert(!_gradient.empty());
    return {_gradient};
  }

  /**
   * Returns the size of the output vector computed by the vertex.
   * TODO: Add support for multiple output dimension with std::tuple
   * and variadic templates.
   */
  virtual constexpr uint32_t getOutputDimension() const = 0;

protected:
  std::vector<float> _output;
  std::vector<float> _gradient;
  virtual std::shared_ptr<Vertex> applyOperation() = 0;

private:
  friend class cereal::access;
  template <typename Archive> void serialize(Archive &archive) {
    (void)archive;
  }
};

} // namespace fortis::comp_graph