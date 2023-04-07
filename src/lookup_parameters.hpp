#pragma once

#include <cereal/access.hpp>
#include <cereal/types/vector.hpp>
#include <memory>
#include <vector>
namespace fortis::parameters {

struct LookupParameter {
  LookupParameter() = default;
  // The constructor takes a variable number of dimensions so that
  // we can, in theory, support arbitrary dimensions. For now
  // the pipeline only supports two dimensions.
  explicit LookupParameter(std::vector<uint32_t>& dimensions)
      : _dimensions(dimensions){};

  std::vector<uint32_t> _dimensions;

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(_dimensions);
  }
};

using LookupParameterPointer = std::unique_ptr<LookupParameter>;
}  // namespace fortis::parameters