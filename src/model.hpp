
#include "lookup_parameters.hpp"
#include "parameters.hpp"
#include <cereal/access.hpp>
#include <memory>

namespace fortis {

using parameters::LookupParameter;
using parameters::LookupParameterPointer;
using parameters::Parameter;
using parameters::ParameterPointer;

class Model : public std::enable_shared_from_this<Model> {

public:
  Model(){};

  explicit Model(ParameterPointer &parameter){};

  ParameterPointer addParameter(const ParameterPointer &parameter);

  LookupParameterPointer
  addLookupParameter(const LookupParameterPointer &lookup_parameter);
  void updateParameterGradients();

  ParameterPointer getParameterByName(const std::string &name);
  LookupParameterPointer getLookupParameterByName(const std::string &name);

  void save(const std::string &file_name) const;
  static std::shared_ptr<Model> load(const std::string &file_name);

private:
  friend class cereal::access;
  template <typename Archive> void serialize(Archive &archive) { archive(); }
};

} // namespace fortis