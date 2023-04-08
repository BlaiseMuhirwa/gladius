
#include <_types/_uint32_t.h>
#include <src/parameters.hpp>
#include <src/trainer.hpp>
#include <stdexcept>

namespace fortis::trainers {

using fortis::parameters::ParameterType;

GradientDescentTrainer::GradientDescentTrainer(std::shared_ptr<Model> model,
                                               float learning_rate)
    : _model(std::move(model)), _learning_rate(learning_rate) {}

void GradientDescentTrainer::takeDescentStep() {
  auto parameters = _model->getParameters();
  for (auto& variant_parameter : parameters) {
    auto parameter = std::get<Parameter>(variant_parameter);
    auto computed_gradient = parameter.getGradient();

    auto parameter_value = parameter.getValue();
    if (computed_gradient.empty()) {
      throw std::runtime_error(
          "Error backpropagating the gradients through the network.");
    }
    if (parameter_value.size() != computed_gradient.size()) {
      throw std::runtime_error(
          "Invalid dimensions for the parameter and "
          "computed gradient. The gradient has size " +
          std::to_string(computed_gradient.size()) +
          " while the parameter has size " +
          std::to_string(parameter_value.size()) + ".");
    }

    if (parameter.getParameterType() == ParameterType::WeightParameter) {
      updateWeightMatrixParameter(parameter_value, computed_gradient);
    } else if (parameter.getParameterType() == ParameterType::BiasParameter) {
      updateBiasVectorParameter(parameter_value.at(0), computed_gradient.at(0));
    } else {
      throw std::runtime_error(
          "Invalid parameter type encountered while "
          "attempting parameter update.");
    }
  }
}

void GradientDescentTrainer::updateWeightMatrixParameter(
    std::vector<std::vector<float>>& weight_matrix,
    std::vector<std::vector<float>>& jacobian) const {
  assert(weight_matrix.size() == jacobian.size());
  assert(weight_matrix.at(0).size() == jacobian.at(0).size());

  for (uint32_t row_index = 0; row_index < weight_matrix.size(); row_index++) {
    for (uint32_t col_index = 0; col_index < weight_matrix.at(0).size();
         col_index++) {
      weight_matrix[row_index][col_index] -=
          _learning_rate * jacobian[row_index][col_index];
    }
  }
}
void GradientDescentTrainer::updateBiasVectorParameter(
    std::vector<float>& bias_vector, std::vector<float>& gradient) const {
  assert(bias_vector.size() == gradient.size());

  for (uint32_t index = 0; index < bias_vector.size(); index++) {
    bias_vector[index] -= _learning_rate * gradient[index];
  }
}

std::shared_ptr<Model> GradientDescentTrainer::getModel() { return _model; }

}  // namespace fortis::trainers