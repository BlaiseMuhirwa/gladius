

#include <memory>
namespace fortis::parameters {

struct LookupParameter {
  LookupParameter(){};
};

using LookupParameterPointer = std::shared_ptr<LookupParameter>;
} // namespace fortis::parameters