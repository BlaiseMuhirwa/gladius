#pragma once

#include "cereal/access.hpp"
#include "utils.hpp"
#include <ios>
#include <memory>
#include <vector>

namespace fortis::parameters {

struct Parameter {
  explicit Parameter(const std::vector<std::vector<float>> &input)
      : _axes(input.size() == 1 ? 1 : 2), _value(input) {}

  constexpr uint32_t axes() const { return _axes; }
  std::vector<std::vector<float>> value() const { return _value; }

private:
  Parameter(){};
  uint32_t _axes;
  std::vector<std::vector<float>> _value;

  friend class cereal::access;

  template <typename Archive> void serialize(Archive &archive) {
    archive(_axes, _value);
  }
};

using ParameterPointer = std::shared_ptr<Parameter>;

} // namespace fortis::parameters