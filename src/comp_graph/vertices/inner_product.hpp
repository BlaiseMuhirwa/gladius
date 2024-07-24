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

namespace gladius::comp_graph {

using gladius::comp_graph::Vertex;
using gladius::comp_graph::VertexPointer;

class InnerProduct final : public Vertex,
                           public std::enable_shared_from_this<InnerProduct> {
 public:
  InnerProduct(VertexPointer left_input, VertexPointer right_input)
      : _left_input(std::move(left_input)),
        _right_input(std::move(right_input)),
        _local_right_gradient(std::nullopt),
        _local_left_gradient(std::nullopt),
        _left_input_jacobian(std::nullopt) {
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

    _output = std::vector<float>(_output_length, 0.F);

    _local_left_gradient = std::vector<float>(
        left_input_shape.first * left_input_shape.second, 0.F);

    _local_right_gradient = std::vector<float>(
        right_input_shape.first * right_input_shape.second, 0.F);
  }

  void forward() final { applyOperation(); }

  void backward(std::optional<std::vector<float>>& upstream_grad) final {
    if (!upstream_grad.has_value()) {
      throw std::runtime_error(
          "Cannot propagate the gradient backward without "
          "setting the upstream gradient first.");
    }
    assert(!_output.empty());

    // Memory for the gradients ought to have been allocated during the forward
    // pass
    assert(_local_left_gradient.has_value());
    assert(_local_right_gradient.has_value());
    assert(upstream_grad.value().size() == _output.size());

    backwardLeftInputImpl(/* upstream_grad = */ upstream_grad.value());
    backwardRightInputImpl(/* upstream_grad = */ upstream_grad.value());
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
  void backwardRightInputImpl(std::vector<float>& upstream_grad) {
    auto left_input_shape = _left_input->getOutputShape();
    assert(upstream_grad.size() == left_input_shape.first);

    auto [num_rows, num_columns] = _right_input->getOutputShape();

    // std::cout << "\t[inner-prod-local-grad-update(right)]" << std::endl;

    for (uint32_t col_index = 0; col_index < num_columns; col_index++) {
      (*_local_right_gradient)[col_index] +=  // NOLINT
          gladius::utils::innerProduct(        // NOLINT
              /* vector = */ upstream_grad,   // NOLINT
              /* matrix = */ _left_input->getOutput(),
              /* col_index = */ col_index);
    }
    // std::cout << "[inner-prod-backward-right-input]" << std::endl;

    _right_input->backward(/* upstream_grad = */ _local_right_gradient);
  }
  /**
   * To get the dimensions of the Jacobian matrix for the local
   * gradient w.r.t the weight matrix, suppose the left input represents
   * a weight matrix parameter W of shape (mxn) and the right input represents
   * a vector x of shape (1xn).
   * Let \Phi be the multiplication operation. Then, it follows that the
   * Jacobian of \Phi w.r.t W has shape (mx(mn)).
   */
  void backwardLeftInputImpl(std::vector<float>& upstream_grad) {
    auto left_input_shape = _left_input->getOutputShape();
    assert(upstream_grad.size() == left_input_shape.first);

    uint32_t total_jacobian_columns =
        left_input_shape.first * left_input_shape.second;
    // std::cout << "\t[inner-prod-starting(left)]" << std::endl;

    if (!_left_input_jacobian.has_value()) {
      // std::cout << "\t[inner-prod-computing jacobian(left)]" << std::endl;

      _left_input_jacobian = std::vector<std::vector<float>>(
          left_input_shape.first,
          std::vector<float>(total_jacobian_columns, 0.F));

#pragma omp parallel for default(none)                                     \
    shared(left_input_shape, total_jacobian_columns, _left_input_jacobian, \
           _right_input)
      for (uint32_t row_index = 0; row_index < left_input_shape.first;
           row_index++) {
        for (uint32_t col_index = 0; col_index < total_jacobian_columns;
             col_index++) {
          uint32_t weight_matrix_col_index =
              col_index % left_input_shape.second;
          uint32_t weight_matrix_row_index =
              (col_index - weight_matrix_col_index) / left_input_shape.second;

          if (row_index == weight_matrix_row_index) {
            (*_left_input_jacobian)[row_index][col_index] =
                _right_input->getOutput().at(0).at(weight_matrix_col_index);
          }
        }
      }
    }
    // std::cout << "\t[inner-prod-local-grad-update(left)]" << std::endl;

    for (uint32_t col_index = 0; col_index < total_jacobian_columns;
         col_index++) {
      (*_local_left_gradient)[col_index] +=  // NOLINT
          gladius::utils::innerProduct(       // NOLINT
              /* vector = */ upstream_grad,
              /* matrix = */ _left_input_jacobian.value(),
              /* col_index = */ col_index);
    }

    // std::cout << "[inner-prod-backward-left-input]" << std::endl;

    _left_input->backward(/* upstream_grad = */ _local_left_gradient);
  }

  std::shared_ptr<Vertex> applyOperation() final {
    auto right_output_vector = _right_input->getOutput().at(0);
    auto size = _left_input->getOutputShape().first;

    for (uint32_t row_index = 0; row_index < size; row_index++) {
      auto current_row = _left_input->getOutput().at(row_index);
      auto inner_product = gladius::utils::innerProduct(
          /*first = */ current_row, /* second = */ right_output_vector);

      _output[row_index] = inner_product;
    }
    return shared_from_this();
  }
  VertexPointer _left_input;
  VertexPointer _right_input;
  std::optional<std::vector<float>> _local_right_gradient;
  std::optional<std::vector<float>> _local_left_gradient;

  std::optional<std::vector<std::vector<float>>> _left_input_jacobian;

  uint32_t _output_length;

  InnerProduct() = default;
  friend class cereal::access;

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Vertex>(this), _left_input, _right_input,
            _output, _local_left_gradient, _local_right_gradient,
            _left_input_jacobian, _output_length);
  }
};

}  // namespace gladius::comp_graph

CEREAL_REGISTER_TYPE(gladius::comp_graph::InnerProduct)