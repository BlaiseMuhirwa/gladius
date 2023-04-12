
#include "trainer.hpp"
#include <_types/_uint32_t.h>
#include <src/params/parameters.hpp>
#include <src/trainers/trainer.hpp>
#include <omp.h>
#include <stdexcept>

namespace fortis::trainers {

using fortis::parameters::ParameterType;

GradientDescentTrainer::GradientDescentTrainer(std::shared_ptr<Model> model,
                                               float learning_rate)
    : _model(std::move(model)), _learning_rate(learning_rate) {}

void GradientDescentTrainer::takeDescentStep() {
  auto parameters = _model->getParameters();
  for (auto& variant_parameter : parameters) {
    auto parameter = std::get<std::shared_ptr<Parameter>>(variant_parameter);
    auto computed_gradient = parameter->getGradient();

    auto parameter_value = parameter->getValue();

    if (computed_gradient.empty()) {
      throw std::runtime_error(
          "Error backpropagating the gradients through the network.");
    }
    uint64_t total_parameter_count = parameter->getParameterCount();
    uint64_t total_gradient_count =
        computed_gradient.size() * computed_gradient.at(0).size();

    if (total_parameter_count != total_gradient_count) {
      throw std::runtime_error(
          "Invalid dimensions for the parameter and "
          "computed gradient. The computed gradient has " +
          std::to_string(total_gradient_count) + " inputs " +
          " while the parameter has " + std::to_string(total_parameter_count) +
          "total inputs.");
    }

    if (parameter->getParameterType() == ParameterType::WeightParameter) {
      updateWeightMatrixParameter(parameter_value, computed_gradient);
    } else if (parameter->getParameterType() == ParameterType::BiasParameter) {
      updateBiasVectorParameter(parameter_value.at(0), computed_gradient.at(0));
    } else {
      throw std::runtime_error(
          "Invalid parameter type encountered while "
          "attempting parameter update.");
    }
  }
}

void GradientDescentTrainer::zeroOutGradients() {
  auto parameters = _model->getParameters();
  for (auto& variant_parameter : parameters) {
    auto parameter = std::get<std::shared_ptr<Parameter>>(variant_parameter);
    parameter->clearGradient();
  }
}

void GradientDescentTrainer::updateWeightMatrixParameter(
    std::vector<std::vector<float>>& weight_matrix,
    std::vector<std::vector<float>>& jacobian) const {
  // assert(weight_matrix.size() == jacobian.size());
  // assert(weight_matrix.at(0).size() == jacobian.at(0).size());

  auto weight_matrix_rows = weight_matrix.size();
  auto weight_matrix_cols = weight_matrix.at(0).size();

  // #pragma omp parallel for default(none)                                      \
//     shared(weight_matrix_rows, weight_matrix_cols, weight_matrix, jacobian, \
//            _learning_rate)
  for (uint32_t row_index = 0; row_index < weight_matrix_rows; row_index++) {
    for (uint32_t col_index = 0; col_index < weight_matrix_cols; col_index++) {
      uint32_t jacobian_index = (row_index * weight_matrix_cols) + col_index;
      weight_matrix[row_index][col_index] -=
          _learning_rate * jacobian[0][jacobian_index];
    }
  }
}
void GradientDescentTrainer::updateBiasVectorParameter(
    std::vector<float>& bias_vector, std::vector<float>& gradient) const {
  assert(bias_vector.size() == gradient.size());

  // #pragma omp parallel for default(none) \
//     shared(_learning_rate, gradient, bias_vector)
  for (uint32_t index = 0; index < bias_vector.size(); index++) {
    bias_vector[index] -= _learning_rate * gradient[index];
  }
}

std::shared_ptr<Model> GradientDescentTrainer::getModel() { return _model; }

}  // namespace fortis::trainers