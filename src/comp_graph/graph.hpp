#pragma once

#include <cstddef>
#include <optional>
#include <set>
#include <src/comp_graph/vertices/vertex.hpp>
#include <stdexcept>
#include <vector>

namespace fortis::comp_graph {

class Graph {
public:
  Graph() : _topologically_sorted_vertices({}), _loss_value(std::nullopt) {}
  Graph(const Graph &) = delete;
  Graph(Graph &&) = delete;

  void addVertex(VertexPointer vertex) {
    _topologically_sorted_vertices.emplace_back(std::move(vertex));
  }

  // TODO: Clean up the vertex interface so that we don't end up with
  //       the following situation (where we are computing the loss value);
  void launchForwardPass() {
    auto graph_size = _topologically_sorted_vertices.size();
    assert(_topologically_sorted_vertices[graph_size - 1]->getName() ==
           "CrossEntropyLoss");

    for (uint32_t vertex_index = 0; vertex_index < graph_size; vertex_index++) {
      auto vertex = _topologically_sorted_vertices[vertex_index];
      vertex->forward();
    }
    auto loss_vertex = _topologically_sorted_vertices[graph_size - 1];
    _loss_value = loss_vertex->getOutput().at(0).at(0);
  }

  void launchBackwardPass() {
    if (!_loss_value.has_value()) {
      throw std::runtime_error(
          "You must compute the value of the loss function first.");
    }
    auto graph_size = _topologically_sorted_vertices.size();
    std::optional<std::vector<std::vector<float>>> previous_gradient =
        std::nullopt;
    for (uint32_t vertex_index = graph_size - 1; vertex_index >= 0;
         vertex_index--) {
      auto vertex = _topologically_sorted_vertices[vertex_index];
      vertex->backward(previous_gradient);
      previous_gradient = vertex->getGradient();
    }
  }

private:
  std::vector<VertexPointer> _topologically_sorted_vertices;
  std::optional<float> _loss_value;
};

} // namespace fortis::comp_graph