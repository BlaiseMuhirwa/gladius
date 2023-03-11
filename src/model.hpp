
#include "lookup_parameters.hpp"
#include "parameters.hpp"
#include <_types/_uint32_t.h>
#include <cereal/access.hpp>
#include <cereal/types/variant.hpp>
#include <cereal/types/vector.hpp>
#include <memory>
#include <variant>

#ifdef RUN_BENCHMARKS
#include <benchmark/benchmark.h>
#endif

namespace fortis {

using parameters::LookupParameter;
using parameters::LookupParameterPointer;
using parameters::Parameter;
using parameters::ParameterPointer;

class Model : public std::enable_shared_from_this<Model> {

public:
  Model(){};

  Parameter &addParameter(uint32_t dimension);

  // For now we assume that we only have 2 dimensions for the vector
  // TODO: Relax this assumption
  Parameter &addParameter(const std::vector<uint32_t> &dimensions);

  LookupParameter &
  addLookupParameter(const LookupParameterPointer &lookup_parameter);

  ParameterPointer getParameterByName(const std::string &name);
  LookupParameterPointer getLookupParameterByName(const std::string &name);

  std::vector<std::variant<Parameter, LookupParameter>> getParameters() const {
    return _parameters;
  }

  void save(const std::string &file_name) const;
  static std::shared_ptr<Model> load(const std::string &file_name);

private:
#ifdef RUN_BENCHMARKS
  static void registerBenchmarkToRun() {
    BENCHMARK(addParameter);
    BENCHMARK(addLookupParameter);
    BENCHMARK(save);
    BENCHMARK(load);
  }

  void launchBenchmarks() { BENCHMARK_MAIN(); }

#endif
  std::vector<std::variant<Parameter, LookupParameter>> _parameters;

  friend class cereal::access;
  template <typename Archive> void serialize(Archive &archive) {
    archive(_parameters);
  }
};

} // namespace fortis