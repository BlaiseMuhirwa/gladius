#pragma once

#include <cereal/access.hpp>
#include <string>

namespace fortis::operations {

class Operation {
  virtual ~Operation() = default;

private:
  friend class cereal::access;
  template <typename Archive> void serialize(Archive &archive) {
    (void)archive;
  }
};

using OperationPointer = std::shared_ptr<Operation>;

} // namespace fortis::operations