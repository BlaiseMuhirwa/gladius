#pragma once

#include <vector>
#include <memory>
#include <cereal/access.hpp>

namespace fortis::parameters {

struct Parameter {
  explicit Parameter(const std::vector<std::vector<float>> &input)
      : _axes(input.size() == 1 ? 1 : 2), _value(input) {}

  constexpr uint32_t axes() const { return _axes; }
  std::vector<std::vector<float>> value() const { return _value; }

private:
  uint32_t _axes;
  std::vector<std::vector<float>> _value;
};

using ParameterPtr = std::shared_ptr<Parameter>;

} // namespace fortis::parameters