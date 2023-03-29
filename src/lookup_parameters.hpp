#pragma once

#include <memory>
#include <src/cereal/access.hpp>
#include <src/cereal/types/vector.hpp>
#include <vector>
namespace fortis::parameters {

struct LookupParameter {
  LookupParameter() {}
  // The constructor takes a variable number of dimensions so that
  // we can, in theory, support arbitrary dimensions. For now
  // the pipeline only supports two dimensions.
  explicit LookupParameter(std::vector<uint32_t> &dimensions)
      : _dimensions(dimensions){};
  
  // Delete the copy constructor and copy assignment operator
  LookupParameter(const LookupParameter &) = delete;
  LookupParameter &operator=(const LookupParameter &) = delete;
  LookupParameter &operator=(LookupParameter &&) = delete;

  std::vector<uint32_t> _dimensions;

  template <typename Archive> void serialize(Archive &archive) {
    archive(_dimensions);
  }
};

using LookupParameterPointer = std::unique_ptr<LookupParameter>;
} // namespace fortis::parameters