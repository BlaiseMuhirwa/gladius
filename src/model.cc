#include <cereal/archives/binary.hpp>
#include "./utils.hpp"
#include "parameters.hpp"
#include <src/model.hpp>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <variant>
namespace fortis {

static inline float const MEAN = 0.F;
static inline float const STD_DEV = 1.F;

Parameter& Model::addParameter(uint32_t dimension) {
  std::random_device random_device;
  std::mt19937 generator(random_device());
  std::normal_distribution<float> distribution(MEAN, STD_DEV);

  std::vector<float> parameter_vector;
  std::generate_n(std::back_inserter(parameter_vector), dimension,
                  [&] { return distribution(generator); });

  auto parameter =
      std::make_shared<Parameter>(Parameter({parameter_vector}));
  _parameters.emplace_back(*parameter);
  return *parameter;
}

// TODO(blaise): Parallelize this implementation with OpenMP
Parameter& Model::addParameter(const std::vector<uint32_t>& dimensions) {
  assert(dimensions.size() == 2);
  auto vector_count = dimensions[0];
  auto per_vector_dimension = dimensions[1];

  std::random_device random_device;
  std::mt19937 generator(random_device());
  std::normal_distribution<float> distribution(MEAN, STD_DEV);

  std::vector<std::vector<float>> parameter_vectors;

  for (uint32_t vec_index = 0; vec_index < vector_count; vec_index++) {
    std::vector<float> current_vector;
    std::generate_n(std::back_inserter(current_vector), per_vector_dimension,
                    [&] { return distribution(generator); });

    parameter_vectors.emplace_back(std::move(current_vector));
  }
  auto parameter =
      std::make_shared<Parameter>(Parameter(std::move(parameter_vectors)));

  _parameters.emplace_back(*parameter);
  return *parameter;
}

// TODO(blaise): Refactor the code below to combine getParameterByID and
// getLookupParameterByID
std::unique_ptr<Parameter> Model::getParameterByID(uint32_t param_id) {
  if (param_id >= _parameters.size()) {
    throw std::invalid_argument(
        "Invalid ID encountered while attempting to access a model parameter.");
  }
  try {
    auto parameter = std::get<Parameter>(_parameters[param_id]);
    return std::make_unique<Parameter>(parameter);
  } catch (const std::bad_variant_access& exception) {
    throw std::invalid_argument(
        "param_id is not compatible with the requested parameter type." +
        std::to_string(*exception.what()));
  }
}

std::unique_ptr<LookupParameter> Model::getLookupParameterByID(
    uint32_t param_id) {
  if (param_id >= _parameters.size()) {
    throw std::invalid_argument(
        "Invalid ID encountered while attempting to access a model parameter.");
  }
  try {
    auto parameter = std::get<LookupParameter>(_parameters[param_id]);
    return std::make_unique<LookupParameter>(parameter);
  } catch (const std::bad_variant_access& exception) {
    throw std::invalid_argument(
        "param_id is not compatible with the requested parameter type." +
        std::to_string(*exception.what()));
  }
}

}  // namespace fortis