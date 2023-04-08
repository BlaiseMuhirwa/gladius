#pragma once

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/variant.hpp>
#include <cereal/types/vector.hpp>
#include <src/lookup_parameters.hpp>
#include <src/parameters.hpp>
#include <memory>
#include <string>
#include <variant>

#ifdef RUN_BENCHMARKS
#include <benchmark/benchmark.h>
#endif

namespace fortis {

using parameters::LookupParameter;
using parameters::LookupParameterPointer;
using parameters::Parameter;
using parameters::ParameterPointer;

class Model {
 public:
  Model() = default;

  /* We want to only have one instance of a model running in the background,
   * which is why we need to delete the copy constructor and copy assignment
   * operator
   */
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  Parameter& addParameter(uint32_t dimension);

  // For now we assume that we only have 2 dimensions for the vector
  // TODO(blaise): Relax this assumption
  Parameter& addParameter(const std::vector<uint32_t>& dimensions);

  LookupParameter& addLookupParameter(
      const LookupParameterPointer& lookup_parameter);

  ParameterPointer getParameterByID(uint32_t param_id);
  LookupParameterPointer getLookupParameterByID(uint32_t param_id);

  std::vector<std::variant<Parameter, LookupParameter>>& getParameters() {
    return _parameters;
  }

  uint32_t getParameterCount() const { return _parameters.size(); }

  void updateParameterGradients();

  void save(const std::string& file_name) const {
    std::ofstream file_stream =
        fortis::utils::handle_ofstream(file_name, std::ios::binary);
    cereal::BinaryOutputArchive output_archive(file_stream);

    output_archive(*this);
  }
  // static std::shared_ptr<Model> load(const std::string &file_name) {
  //   std::ifstream file_stream =
  //       fortis::handle_ifstream(file_name, std::ios::binary);

  //   cereal::BinaryInputArchive input_archive(file_stream);
  //   // Model model;
  //   // input_archive(model);
  //   // return std::make_shared<Model>(model);
  //   std::shared_ptr<Model> deserialized_model(new Model());
  //   input_archive(*deserialized_model);

  //   return nullptr;
  // }

 private:
  std::vector<std::variant<Parameter, LookupParameter>> _parameters;

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(_parameters);
  }
};

}  // namespace fortis