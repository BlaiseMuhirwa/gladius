
#include "cereal/access.hpp"
#include "parameters.hpp"
#include <memory>

namespace fortis {

using parameters::Parameter;
using parameters::ParameterPointer;

class Model : public std::enable_shared_from_this<Model> {
  Model(){};

  explicit Model(ParameterPointer &parameter){};

  void addParameter(const ParameterPointer &parameter);

private:
  friend class cereal::access;
  template <typename Archive> void serialize(Archive &archive) { archive(); }
};

} // namespace fortis