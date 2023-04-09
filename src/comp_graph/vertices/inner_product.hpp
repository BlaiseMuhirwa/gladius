#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <_types/_uint32_t.h>
#include <src/comp_graph/vertices/vertex.hpp>
#include <src/utils.hpp>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fortis::comp_graph {

using fortis::comp_graph::Vertex;
using fortis::comp_graph::VertexPointer;

class InnerProduct final : public Vertex,
                           public std::enable_shared_from_this<InnerProduct> {
 public:
  InnerProduct(VertexPointer left_input, VertexPointer right_input)
      : _left_input(std::move(left_input)),
        _right_input(std::move(right_input)) {
    auto left_input_shape = _left_input->getOutputShape();
    auto right_input_shape = _right_input->getOutputShape();

    if (!(left_input_shape.second == right_input_shape.second)) {
      throw std::invalid_argument(
          "Dimension mismatch for the inputs to InnerProduct vertex. Make sure "
          "that the two inputs have the same dimensions.");
    }
    // We treat the operation as representing Matrix-vector multiplication if
    // the first dimension mismatch. Otherwise, the operation represents the
    // inner product of two vectors.
    // For instance, we can have z = Wx for W with shape (n,d) and x with shape
    // (1, d) or we can have z = u^Tv for u, v with shape (1, d)
    if (left_input_shape.first != right_input_shape.first) {
      _output_length = left_input_shape.first;
    } else {
      _output_length = left_input_shape.second;
    }
    // std::cout << "[mult] -- allocating size " << _output_length << std::endl;
    _output.reserve(_output_length);
  }

  void forward() final {
    assert(_output.empty());
    applyOperation();
  }

  void backward() final {
    if (!_upstream_gradient.has_value()) {
      throw std::runtime_error(
          "Cannot propagate the gradient backward without "
          "setting the upstream gradient first.");
    }
    assert(!_output.empty());
    assert(_upstream_gradient.value().size() == 1);
    assert(_upstream_gradient.value().at(0).size() == _output.size());

    backwardLeftInputImpl();
    backwardRightInputImpl();
    // std::cout << "[inner-prod-finished upstream grads updates]" << std::endl;
  }

  inline std::string getName() final { return "InnerProduct"; }

  std::pair<uint32_t, uint32_t> getOutputShape() const final {
    return std::make_pair(1, _output_length);
  }

 private:
  /**
   * Note: For now we are assuming that the right input represents
   * some vector (for example, a computation result from ReLU).
   * With this assumption in mind, it follows directly that the
   * Jacobian of the multiplication operation w.r.t the right input
   * is precisely the left input.
   */
  void backwardRightInputImpl() {
    auto [row_size, col_size] = _right_input->getOutputShape();
    // Checks if this is the first time backpropagating through this vertex
    // On the first pass we populate the partial derivatives
    if (_local_right_gradient.empty()) {
      _local_right_gradient =
          std::vector<std::vector<float>>(1, std::vector<float>(col_size, 0.F));
    }
    auto num_columns = _left_input->getOutputShape().second;
    for (uint32_t col_index = 0; col_index < num_columns; col_index++) {
      _local_right_gradient[0][col_index] += fortis::utils::innerProduct(
          /* vector = */ _upstream_gradient.value().at(0),  // NOLINT
          /* matrix = */ _left_input->getOutput(), /* col_index = */ col_index);
    }

    _right_input->setUpstreamGradient(/* gradient = */ _local_right_gradient);
  }
  /**
   * To get the dimensions of the Jacobian matrix for the local
   * gradient w.r.t the weight matrix, suppose the left input represents
   * a weight matrix parameter W of shape (mxn) and the right input represents
   * a vector x of shape (1xn).
   * Let \Phi be the multiplication operation. Then, it follows that the
   * Jacobian of \Phi w.r.t W has shape (mx(mn)).
   */
  void backwardLeftInputImpl() {
    auto right_input_size = _right_input->getOutputShape().second;
    auto num_rows = _left_input->getOutputShape().first;
    auto num_columns = num_rows * right_input_size;

    std::vector<std::vector<float>> jacobian_matrix(
        num_rows, std::vector<float>(num_columns, 0.F));

    for (uint32_t row_index = 0; row_index < num_rows; row_index++) {
      for (uint32_t col_index = 0; col_index < num_columns; col_index++) {
        uint32_t weight_matrix_row_index = col_index / num_columns;
        uint32_t weight_matrix_col_index = col_index % right_input_size;

        if (row_index == weight_matrix_row_index) {
          jacobian_matrix[row_index][col_index] =
              _right_input->getOutput().at(0).at(weight_matrix_col_index);
        }
      }
    }

    auto [row_size, col_size] = _left_input->getOutputShape();
    // Checks if this is the first time backpropagating through this vertex
    // On the first pass we populate the partial derivatives
    if (_local_left_gradient.empty()) {
      _local_left_gradient = std::vector<std::vector<float>>(
          1, std::vector<float>(row_size * col_size, 0.F));
    }

    for (uint32_t col_index = 0; col_index < (row_size * col_size);
         col_index++) {
      _local_left_gradient[0][col_index] += fortis::utils::innerProduct(
          /* vector = */ _upstream_gradient.value().at(0),  // NOLINT
          /* matrix = */ jacobian_matrix,
          /* col_index = */ col_index);
    }
    _left_input->setUpstreamGradient(/* gradient = */ _local_left_gradient);
  }

  std::shared_ptr<Vertex> applyOperation() final {
    auto right_output_vector = _right_input->getOutput().at(0);
    auto size = _left_input->getOutputShape().first;

    for (uint32_t row_index = 0; row_index < size; row_index++) {
      auto current_row = _left_input->getOutput().at(row_index);
      auto inner_product = fortis::utils::innerProduct(
          /*first = */ current_row, /* second = */ right_output_vector);

      _output.emplace_back(inner_product);
    }
    return shared_from_this();
  }
  VertexPointer _left_input;
  VertexPointer _right_input;
  std::vector<std::vector<float>> _local_left_gradient;
  std::vector<std::vector<float>> _local_right_gradient;
  uint32_t _output_length;

  InnerProduct() = default;
  friend class cereal::access;

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Vertex>(this), _left_input, _right_input,
            _output, _local_left_gradient, _local_right_gradient,
            _upstream_gradient, _output_length);
  }
};

}  // namespace fortis::comp_graph

CEREAL_REGISTER_TYPE(fortis::comp_graph::InnerProduct)