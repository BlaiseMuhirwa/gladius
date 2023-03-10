

#include <memory>
namespace fortis::parameters {

struct LookupParameter {
  LookupParameter(){};
};

using LookupParameterPointer = std::unique_ptr<LookupParameter>;
} // namespace fortis::parameters