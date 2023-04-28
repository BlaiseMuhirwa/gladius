#pragma once

#include <src/comp_graph/vertices/activ_functions.hpp>
#include <src/comp_graph/vertices/vertex.hpp>
#include <cstddef>
#include <omp.h>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

namespace fortis::comp_graph {

class Graph {
 public:
  Graph() : _topologically_sorted_vertices({}), _loss_value(std::nullopt) {}
  Graph(const Graph&) = delete;
  Graph(Graph&&) = delete;
  Graph& operator=(const Graph&) = delete;
  Graph& operator=(Graph&&) = delete;

  inline void clearComputationGraph() {
    if (!_topologically_sorted_vertices.empty()) {
      // for (auto& vertex : _topologically_sorted_vertices) {
      //   vertex->zeroOutGradients();
      // }
      _topologically_sorted_vertices.clear();
    }
    if (_loss_value) {
      _loss_value = std::nullopt;
    }
  }

  uint32_t getVerticesCount() const {
    return _topologically_sorted_vertices.size();
  }

  VertexPointer getVertexAtIndex(uint32_t vertex_id) {
    if (vertex_id < getVerticesCount()) {
      return _topologically_sorted_vertices[vertex_id];
    }
    return nullptr;
  }

  void addVertex(VertexPointer vertex) {
    _topologically_sorted_vertices.emplace_back(std::move(vertex));
  }

  /**
   * Returns a tuple of the predicted label and the loss value
   * TODO: Clean up the vertex interface so that we don't end up with
   *       the following situation (where we are computing the loss value).
   */
  std::tuple<uint32_t, float> launchForwardPass() {
    auto graph_size = _topologically_sorted_vertices.size();
    assert(_topologically_sorted_vertices[graph_size - 1]->getName() ==
           "CrossEntropyLoss");

    uint32_t prediction;
    for (uint32_t vertex_index = 0; vertex_index < graph_size; vertex_index++) {
      auto vertex = _topologically_sorted_vertices[vertex_index];
      vertex->forward();
      if (vertex->getName() == "SoftMax") {
        prediction =
            dynamic_cast<fortis::comp_graph::SoftMaxActivation*>(vertex.get())
                ->getPredictedLabel();
      }
    }
    auto loss_vertex = _topologically_sorted_vertices[graph_size - 1];
    assert(loss_vertex->getName() == "CrossEntropyLoss");

    _loss_value = loss_vertex->getOutput().at(0).at(0);
    return {prediction, _loss_value.value()};
  }

  // void launchBackwardPass() {
  //   if (!_loss_value.has_value()) {
  //     throw std::runtime_error(
  //         "You must compute the value of the loss function first.");
  //   }
  //   auto graph_size = _topologically_sorted_vertices.size();

  //   for (int vertex_index = graph_size - 1; vertex_index >= 0;
  //   vertex_index--) {
  //     auto vertex = _topologically_sorted_vertices[vertex_index];
  //     vertex->backward();
  //   }
  // }

 private:
  std::vector<VertexPointer> _topologically_sorted_vertices;
  std::optional<float> _loss_value;
};

}  // namespace fortis::comp_graph