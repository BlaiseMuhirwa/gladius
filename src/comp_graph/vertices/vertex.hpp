#pragma once

#include <cereal/access.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include <cassert>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gladius::comp_graph {

// class Vertex;
// using VertexPointer = std::shared_ptr<Vertex>;

// struct Expression {
//   void operator()(VertexPointer& vertex) { _value = std::move(vertex); }
//   VertexPointer _value;

//   Expression() {}
//   friend class cereal::access;

//   template <typename Archive>
//   void serialize(Archive& archive) {
//     archive(_value);
//   }
// };

class Vertex {
 public:
  Vertex() = default;
  /* Move constructor */
  Vertex(Vertex&& other) noexcept
      : _local_gradient(std::move(other._local_gradient)),
        _output(std::move(other._output)) {}

  /* Move assignment operator */

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
  virtual void backward(std::optional<std::vector<float>>& upstream_grad) = 0;

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
   *
   * TODO: Change this function to `updateUpstreamGradient`. This implementation
   *       limits the types of models we can have. For instance, we can't have
   *       backpropagation through a vertex multiple times, which is something
   *       we really want. An easy fix would be to always set _upstream_gradient
   *       to the provided gradient parameter, but there are a few more checks
   *       that should be taken care of first.
  */
  // inline void setUpstreamGradient(std::vector<std::vector<float>>& gradient)
  // {
  //   if (_upstream_gradient.has_value()) {
  //     throw std::runtime_error("The upstream gradient for vertex " +
  //     getName() +
  //                              " has already been set.");
  //   }
  //   // TODO(blaise): Come up with a better way to update the gradient
  //   //  without doing this copy. We can, for instance, initialize the
  //   //  upstream gradient, and then just add to it
  //   _upstream_gradient = gradient;
  // }

  inline void zeroOutGradients() {
    if (_local_gradient.has_value()) {
      _local_gradient = std::nullopt;
    }
  }

  /**
   * Returns the output of the forward pass through the vertex.
   */
  virtual inline std::vector<float>& getOutput() {
    assert(!_output.empty());
    return _output;
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
  virtual inline std::vector<float>& getGradient() {
    assert(_local_gradient.has_value());
    return _local_gradient.value();
  }

  /**
   * Returns the shape of the output vector computed by the vertex.
   */
  virtual std::pair<uint32_t, uint32_t> getOutputShape() const = 0;

 protected:
  /**
   * Applies the main operation implemented by the vertex.
   * For instance, if the vertex computes the expression
   * z = Ax + b, this method is responsible for implementing
   * this addition operation.
   */
  virtual std::shared_ptr<Vertex> applyOperation() = 0;

  std::optional<std::vector<float>> _local_gradient;
  std::vector<float> _output;

 private:
  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(_local_gradient);
  }
};
using VertexPointer = std::shared_ptr<Vertex>;

}  // namespace gladius::comp_graph