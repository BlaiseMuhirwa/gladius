#pragma once

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/variant.hpp>
#include <cereal/types/vector.hpp>
#include <src/params/lookup_parameters.hpp>
#include <src/params/parameters.hpp>
#include <memory>
#include <string>
#include <variant>

#ifdef RUN_BENCHMARKS
#include <benchmark/benchmark.h>
#endif

namespace fortis {

using parameters::LookupParameter;
using parameters::Parameter;

class Model {
 public:
  Model() = default;

  /* We want to only have one instance of a model running in the background,
   * which is why we need to delete the copy constructor and copy assignment
   * operator
   * TODO: Have the model hold an optional graph object and define
   * Model::operator() which will launch the forward pass with the given input
   * sample.
   */
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  // Parameter& addParameter(uint32_t dimension);

  // For now we assume that we only have 2 dimensions for the vector
  // TODO(blaise): Relax this assumption
  void addParameter(const std::vector<uint32_t>&& dimensions);

  void addLookupParameter(LookupParameter& lookup_parameter);

  std::shared_ptr<Parameter> getParameterByID(uint32_t param_id);
  std::shared_ptr<LookupParameter> getLookupParameterByID(uint32_t param_id);

  std::vector<std::shared_ptr<Parameter>>& getParameters() {
    return _parameters;
  }

  uint32_t getParameterCount() const { return _parameters.size(); }

  void updateParameterGradients();

  // void save(const std::string& file_name) const {
  //   std::ofstream file_stream =
  //       fortis::utils::handle_ofstream(file_name, std::ios::binary);
  //   cereal::BinaryOutputArchive output_archive(file_stream);

  //   output_archive(*this);
  // }
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
  std::vector<std::shared_ptr<Parameter>> _parameters;

  // friend class cereal::access;
  // template <typename Archive>
  // void serialize(Archive& archive) {
  //   archive(_parameters);
  // }
};

}  // namespace fortis