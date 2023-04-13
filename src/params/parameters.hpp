#pragma once

#include <cereal/access.hpp>
#include <src/utils.hpp>
#include <_types/_uint32_t.h>
#include <ios>
#include <memory>
#include <stdexcept>
#include <vector>

namespace fortis::parameters {

enum class ParameterType { WeightParameter, BiasParameter };

struct Parameter {
  explicit Parameter(std::vector<std::vector<float>>&& input)
      : _value(std::move(input)) {
    if (_value.empty()) {
      throw std::invalid_argument(
          "Fortis parameter initialization requires a non-empty vector(s).");
    }
    auto total_parameters = getParameterCount();
    // _gradient = std::vector<std::vector<float>>(
    //     1, std::vector<float>(total_parameters, 0.F));
  }

  Parameter(const Parameter&) = delete;
  Parameter& operator=(const Parameter&) = delete;

  std::vector<std::vector<float>> getValue() const { return _value; }

  std::vector<std::vector<float>> getGradient() const { return _gradient; }

  inline void clearGradient() {
    if (!_gradient.empty()) {
      _gradient.clear();
    }
  }

  /**
   * We delegate all input validations to the parameter vertex, which is purely
   * a wrapper around instances of this class. In this spirit, we need not
   * check again if the gradient is properly formatter or not in this class.
   */
  void updateGradient(std::vector<std::vector<float>>& gradient) {
    // assert(gradient.size() == _gradient.size());
    // assert(_gradient.at(0).size() == gradient.at(0).size());

    // auto row_count = gradient.size();
    // auto column_count = gradient.at(0).size();
    // for (uint64_t row_index = 0; row_index < row_count; row_index++) {
    //   for (uint64_t col_index = 0; col_index < column_count; col_index++) {
    //     _gradient[row_index][col_index] += gradient[row_index][col_index];
    //   }
    // }
    _gradient = gradient;
  }

  /**
   * Returns the total number of trainable parameters. For instance,
   * If the parameter wraps a matrix of mxn dimensions, the total
   * number of parameters is mxn
   */
  inline uint64_t getParameterCount() const {
    return _value.size() * _value.at(0).size();
  }

  ParameterType getParameterType() {
    auto dimension = _value.size();
    if (dimension == 1) {
      return ParameterType::BiasParameter;
    }
    return ParameterType::WeightParameter;
  }

  std::pair<uint32_t, uint32_t> getParameterShape() const {
    return {_value.size(), _value[0].size()};
  }

 private:
  Parameter(){};
  std::vector<std::vector<float>> _value;
  std::vector<std::vector<float>> _gradient;

  friend class cereal::access;

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(_value, _gradient);
  }
};

}  // namespace fortis::parameters