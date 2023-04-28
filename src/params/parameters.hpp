#pragma once

#include <cereal/access.hpp>
#include <_types/_uint32_t.h>
#include <src/utils.hpp>
#include <ios>
#include <memory>
#include <stdexcept>
#include <vector>

namespace fortis::parameters {

enum class ParameterType { WeightParameter, BiasParameter };

class Parameter {
 public:
  explicit Parameter(std::vector<std::vector<float>>&& input)
      : _value(std::move(input)) {
    if (_value.empty()) {
      throw std::invalid_argument(
          "Fortis parameter initialization requires a non-empty vector(s).");
    }
    auto total_parameters = getParameterCount();
    _gradient = std::vector<float>(total_parameters, 0.F);
  }

  Parameter(const Parameter&) = delete;
  Parameter& operator=(const Parameter&) = delete;

  Parameter(Parameter&&) = delete;
  Parameter& operator=(const Parameter&&) = delete;

  std::vector<std::vector<float>>& getValue() { return _value; }

  std::vector<float>& getGradient() { return _gradient; }

  inline void zeroOutGradient() {
    if (!_gradients_zeroed_out) {
      for (float& grad : _gradient) {
        grad = 0.F;
      }
    }
    _gradients_zeroed_out = true;
  }

  /**
   * We delegate all input validations to the parameter vertex, which is purely
   * a wrapper around instances of this class. In this spirit, we need not
   * check again if the gradient is properly formatter or not in this class.
   */
  void updateGradient(std::vector<float>& gradient) {
    assert(gradient.size() == _gradient.size());

    auto size = gradient.size();
#pragma omp parallel for default(none) shared(_gradient, gradient, size)
    for (uint64_t index = 0; index < size; index++) {
      _gradient[index] = gradient[index];
    }
    _gradients_zeroed_out = false;

    // std::cout << "[parameter-finished-grad-upate]" << std::endl;
    // _gradient = gradient;
  }

  /**
   * Returns the total number of trainable parameters. For instance,
   * If the parameter wraps a matrix of mxn dimensions, the total
   * number of parameters is mxn
   */
  inline uint64_t getParameterCount() const {
    return _value.size() * _value.at(0).size();
  }

  ParameterType getParameterType() const {
    auto dimension = _value.size();
    if (dimension == 1) {
      return ParameterType::BiasParameter;
    }
    return ParameterType::WeightParameter;
  }

  std::pair<uint32_t, uint32_t> getParameterShape() const {
    return {_value.size(), _value[0].size()};
  }

  void updateParameterValue(float update_factor = 0.0) {
    auto total_rows = _value.size();
    auto total_cols = _value.at(0).size();

    for (uint64_t row = 0; row < total_rows; row++) {
      for (uint64_t col = 0; col < total_cols; col++) {
        uint64_t grad_index = (row * total_cols) + col;
        _value[row][col] += (update_factor * _gradient[grad_index]);
      }
    }
  }

  void printValue() const {
    auto param_type = getParameterType();
    if (param_type != ParameterType::BiasParameter) {
      return;
    }

    for (const auto& val : _value.at(0)) {
      std::cout << val << " ";
    }
    std::cout << "\n";
  }

 private:
  Parameter(){};
  std::vector<std::vector<float>> _value;
  std::vector<float> _gradient;

  bool _gradients_zeroed_out = true;

  friend class cereal::access;

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(_value, _gradient, _gradients_zeroed_out);
  }
};

}  // namespace fortis::parameters