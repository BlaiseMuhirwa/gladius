#pragma once

#include <vector>

namespace fortis::parameters {

struct Parameter {
  explicit Parameter(const std::vector<std::vector<float>> &input)
      : _axes(input.size() == 1 ? 1 : 2), _value(input) {}

  constexpr uint32_t axes() const { return _axes; }
  std::vector<std::vector<float>> value() const { return _value; }
  throw;

private:
  uint32_t _axes;
  std::vector<std::vector<float>> _value;
};

} // namespace fortis::parameters