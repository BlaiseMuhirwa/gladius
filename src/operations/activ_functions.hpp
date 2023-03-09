
#include "base_op.hpp"
#include <cassert>
#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cstddef>
#include <format>
#include <math.h>
#include <memory>
#include <stdexcept>

namespace fortis {
using fortis::Expression;
using fortis::Vertex;

class TanHActivation : public Vertex,
                       public std::enable_shared_from_this<TanHActivation> {
  /**
   * The TanH activation computes the hyperbolic tangent function
   * We recall that for any input value x, TanH(x) is given by
   *
   *          TanH(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}
   **/
public:
  explicit TanHActivation(std::vector<VertexPointer> &incoming_edges)
      : _incoming_edges(incoming_edges) {
    if (_incoming_edges.size() != 1) {
      throw std::runtime_error(
          std::format("TanH activation function expects a single vector as "
                      "input. Received {} vectors",
                      std::to_string(_incoming_edges.size())));
    }
  }

  void forward() final {
    assert(_incoming_edges.size() != 0);
    assert(_output.size() == 0);
    applyOperation();
  }

  /**
   * The local gradient for TanH is expressed as follows:
   * - numerator: 4 * exp(-2x)
   * - denomiator: (1+exp(-2x))^2
   * simplifying, we get that d(tanh(x))/dx = 1 - [tanh(x)^2]
   *
   */
  void backward() final {}

private:
  std::shared_ptr<Vertex> applyOperation() final {

    auto incoming_edge = _incoming_edges[0];
    auto input_vector = incoming_edge->getOuput()[0];
    auto size = input_vector.size();

    for (uint32_t neuron_index = 0; neuron_index < size; neuron_index++) {
      float exponential_term = exp(-2 * input_vector[neuron_index]);
      float tanh_activation = (1 - exponential_term) / (1 + exponential_term);
      _output.push_back(tanh_activation);
    }
    return shared_from_this();
  }

  std::vector<VertexPointer> _incoming_edges;
  std::vector<float> _output;

  friend class cereal::access;
  template <typename Archive> void serialize(Archive &archive) {
    archive(cereal::base_class<Vertex>(this), _incoming_edges, _output);
  }
};

} // namespace fortis

CEREAL_REGISTER_TYPE(fortis::Vertex)
