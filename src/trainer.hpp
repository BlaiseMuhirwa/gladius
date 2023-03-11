
#include "model.hpp"
#include <cereal/access.hpp>
#include <memory>
#include <stdexcept>

namespace fortis::trainers {

class GradientDescentTrainer {
public:
  GradientDescentTrainer(std::shared_ptr<Model> &model, float learning_rate)
      : _model(std::move(model)), _learning_rate(learning_rate) {}

  void takeDescentStep() {
    auto parameters = _model->getParameters();
    for (auto &parameter : parameters) {
      auto computed_gradient = parameter->getGradient().at(0);
      auto parameter_value = parameter->getValue();
      if (computed_gradient.empty()) {
        throw std::runtime_error(
            "Error backpropagating the gradients through the network.");
      }
      if (parameter_value.size() != computed_gradient.size()) {
        throw std::runtime_error("Invalid dimensions for the parameter and "
                                 "computed gradient. The gradient has size " +
                                 std::to_string(computed_gradient.size()) +
                                 " while the parameter has size " +
                                 std::to_string(parameter_value.size()) + ".");
      }
      std::for_each(computed_gradient.begin(), computed_gradient.end(),
                    [&](float partial_derivative) {
                      partial_derivative *= _learning_rate;
                    });
      auto parameter_size = parameter_value.size();
      for (uint32_t index = 0; index < parameter_size; index++) {
        parameter_value[index] -= _learning_rate * computed_gradient[index];
      }
    }
  }
  std::shared_ptr<Model> getModel() const { return _model; }

private:
  std::shared_ptr<Model> _model;
  float _learning_rate;

  friend class cereal::access;
  template <typename Archive> void serialize(Archive &archive) {
    archive(_model, _learning_rate);
  }
};

} // namespace fortis::trainers