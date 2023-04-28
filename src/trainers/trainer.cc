
#include "trainer.hpp"
#include <_types/_uint32_t.h>
#include <src/params/parameters.hpp>
#include <src/trainers/trainer.hpp>
#include <omp.h>
#include <stdexcept>

namespace fortis::trainers {

GradientDescentTrainer::GradientDescentTrainer(std::shared_ptr<Model> model,
                                               float learning_rate)
    : _model(std::move(model)), _learning_rate(learning_rate) {}

void GradientDescentTrainer::takeDescentStep() {
  for (auto& parameter : _model->getParameters()) {
    auto computed_gradient = parameter->getGradient();

    auto parameter_value = parameter->getValue();

    if (computed_gradient.empty()) {
      throw std::runtime_error(
          "Error backpropagating the gradients through the network.");
    }
    uint64_t total_parameter_count = parameter->getParameterCount();
    uint64_t total_gradient_count = computed_gradient.size();

    if (total_parameter_count != total_gradient_count) {
      throw std::runtime_error(
          "Invalid dimensions for the parameter and "
          "computed gradient. The computed gradient has " +
          std::to_string(total_gradient_count) + " inputs " +
          " while the parameter has " + std::to_string(total_parameter_count) +
          "total inputs.");
    }

    parameter->updateParameterValue(
        /* update_factor = */ (-1 * _learning_rate));

  }
}

void GradientDescentTrainer::zeroOutGradients() {
  for (auto& parameter : _model->getParameters()) {
    parameter->zeroOutGradient();
  }
}


std::shared_ptr<Model> GradientDescentTrainer::getModel() { return _model; }

}  // namespace fortis::trainers