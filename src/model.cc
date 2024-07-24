#include <cereal/archives/binary.hpp>
#include <_types/_uint32_t.h>
#include <src/model.hpp>
#include <src/params/parameters.hpp>
#include <src/utils.hpp>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <variant>
namespace gladius {

// TODO(blaise): Parallelize this implementation with OpenMP
void Model::addParameter(const std::vector<uint32_t>&& dimensions) {
  assert(!dimensions.empty());

  std::optional<uint32_t> vector_count = std::nullopt;
  if (dimensions.size() == 2) {
    vector_count = dimensions[0];
  }
  uint32_t vector_dimension =
      vector_count.has_value() ? dimensions[1] : dimensions[0];

  if (!vector_count.has_value()) {
    // Initialize the bias parameter
    std::vector<float> bias(vector_dimension, 0.F);
    _parameters.emplace_back(new Parameter({bias}));
    return;
  }
  // Initialize the weight parameter
  std::random_device random_device;
  std::mt19937 generator(random_device());

  float xavier_variance = 1.F / (dimensions[1]);
  float he_variance_initialization = 2 * xavier_variance;

  std::normal_distribution<float> distribution(
      0.F, sqrt(he_variance_initialization));

  std::vector<std::vector<float>> parameter_vectors;

  for (uint32_t vec_index = 0; vec_index < vector_count.value(); vec_index++) {
    std::vector<float> current_vector;
    std::generate_n(std::back_inserter(current_vector), dimensions[1],
                    [&] { return distribution(generator); });

    parameter_vectors.push_back(std::move(current_vector));
  }

  _parameters.emplace_back(new Parameter(std::move(parameter_vectors)));

}

// TODO(blaise): Refactor the code below to combine getParameterByID and
// getLookupParameterByID
std::shared_ptr<parameters::Parameter> Model::getParameterByID(
    uint32_t param_id) {
  if (param_id >= _parameters.size()) {
    throw std::invalid_argument(
        "Invalid ID encountered while attempting to access a model parameter.");
  }

  auto parameter = _parameters[param_id];
  return parameter;
  // try {
  //   auto parameter =
  //       std::get<std::shared_ptr<Parameter>>(_parameters[param_id]);
  //   return parameter;
  // } catch (const std::bad_variant_access& exception) {
  //   throw std::invalid_argument(
  //       "param_id is not compatible with the requested parameter type." +
  //       std::to_string(*exception.what()));
  // }
}

// std::shared_ptr<LookupParameter> Model::getLookupParameterByID(
//     uint32_t param_id) {
//   if (param_id >= _parameters.size()) {
//     throw std::invalid_argument(
//         "Invalid ID encountered while attempting to access a model
//         parameter.");
//   }
//   try {
//     auto parameter =
//         std::get<std::shared_ptr<LookupParameter>>(_parameters[param_id]);
//     return parameter;
//   } catch (const std::bad_variant_access& exception) {
//     throw std::invalid_argument(
//         "param_id is not compatible with the requested parameter type." +
//         std::to_string(*exception.what()));
//   }
// }

}  // namespace gladius