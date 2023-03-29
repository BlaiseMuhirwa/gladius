#pragma once

#include <cereal/access.hpp>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
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
   * input forward in the DAG.
   */
  virtual void forward() = 0;

  /**
   * The backward function is responsible for computing local
   * gradients at a specific vertex and updating the gradient
   * of the loss function with respect to the given parameter
   * via the chain rule.
   */
  virtual void backward() = 0;

  /**
   * While this method is indeed part of the mechanics of the backward
   * computation, there are a few good reasons for having it as a separate
   * piece of logic.

   * 1. For one, since, at the core, vertices represent unique operations
   *    in the computation graph, it is possible (and will often happen)
   *    that one vertex has more than one parent (i.e., the vertex in question
   *    propagates backwards to more than one other vertex). For example,
   *    in the expression, z = Ax + b, the vertex representing the addition
   *    operation has parents "Ax" and "b".
   *    In such cases, this method ensures that the gradient with respect to
   *    the proper variable is propagated backwards.
   *
   * 2. Secondly, this allows us to also have a clear implementation logic.
   *    Instead of having a "black-boxy" backward method that is responsible
   *    for handling everything during the backward computation, we opt to
   *    have this method so that it is clear what steps must be completed
   *    before the backward computations continue unravelling.
  */
  void inline setUpstreamGradient(std::vector<std::vector<float>> &gradient) {
    if (_upstream_gradient.has_value()) {
      throw std::runtime_error("The upstream gradient for vertex " + getName() +
                               " has already been set.");
    }
    _upstream_gradient = std::move(gradient);
  }

  /**
   * Returns the output of the forward pass through the vertex.
   */
  virtual inline std::vector<std::vector<float>> getOutput() const {
    assert(!_output.empty());
    return {_output};
  }

  /**
   * Returns the name of the corresponding operation implemented by
   * the vertex.
   */
  virtual inline std::string getName() = 0;

  /**
   * Returns the gradient of the loss function with respect to
   * the operation computed by the vertex.
   */
  virtual inline std::vector<std::vector<float>> getGradient() const {
    assert(!_local_gradient.empty());
    return _local_gradient;
  }

  /**
   * Returns the shape of the output vector computed by the vertex.
   */
  virtual std::pair<uint32_t, uint32_t> getOutputShape() const = 0;

protected:
  std::vector<float> _output;

  /**
   * Applies the main operation implemented by the vertex.
   * For instance, if the vertex computes the expression
   * z = Ax + b, this method is responsible for implementing
   * this addition operation.
   */
  virtual std::shared_ptr<Vertex> applyOperation() = 0;

  // This is an optional because we want the upstream gradient to be null
  // for any specific vertex until the backward computation arrives
  // at it.
  std::optional<std::vector<std::vector<float>>> _upstream_gradient;
  std::vector<std::vector<float>> _local_gradient;

private:
  friend class cereal::access;
  template <typename Archive> void serialize(Archive &archive) {
    (void)archive;
  }
};

} // namespace fortis::comp_graph