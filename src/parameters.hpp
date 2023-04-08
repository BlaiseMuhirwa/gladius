#pragma once

#include <cereal/access.hpp>
#include "utils.hpp"
#include <_types/_uint32_t.h>
#include <ios>
#include <memory>
#include <stdexcept>
#include <vector>

namespace fortis::parameters {

enum class ParameterType { WeightParameter, BiasParameter };

struct Parameter {
  explicit Parameter(std::vector<std::vector<float>>&& input) {
    if (input.empty()) {
      throw std::invalid_argument(
          "Fortis parameter initialization requires a non-empty vector(s).");
    }
    _value = std::move(input);
  }

  /**
   * TODO: Delete the copy constructor and copy assignment operator
   *   Parameter(const Parameter &) = delete;
   *   Parameter &operator=(const Parameter &) = delete;
   *   Parameter &operator=(Parameter &&) = delete;
   */
  Parameter(const Parameter&) = delete;
  Parameter& operator=(const Parameter&) = delete;

  std::vector<std::vector<float>> getValue() const { return _value; }

  std::vector<std::vector<float>> getGradient() const { return _gradient; }

  void setGradient(std::vector<std::vector<float>>& gradient) {
    _gradient = std::move(gradient);
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