#include "model.hpp"
#include "./utils.hpp"
#include "cereal/archives/binary.hpp"
#include "parameters.hpp"
#include <algorithm>
#include <iterator>
#include <random>
namespace fortis {

static inline float const MEAN = 0.0f;
static inline float const STD_DEV = 1.0f;

Parameter &Model::addParameter(uint32_t dimension) {
  std::random_device random_device;
  std::mt19937 generator(random_device());
  std::normal_distribution<float> distribution(MEAN, STD_DEV);

  std::vector<float> parameter_vector;
  std::generate_n(std::back_inserter(parameter_vector), dimension,
                  [&] { return distribution(generator); });

  auto parameter =
      std::shared_ptr<Parameter>(new Parameter({parameter_vector}));
  return *parameter;
}

// TODO: Parallelize this implementation with OpenMP
Parameter &Model::addParameter(const std::vector<uint32_t> &dimensions) {
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
      std::shared_ptr<Parameter>(new Parameter(std::move(parameter_vectors)));

  return *parameter;
}

void Model::save(const std::string &file_name) const {
  std::ofstream file_stream = fortis::handle_ofstream(file_name, std::ios::binary);
  cereal::BinaryOutputArchive output_archive(file_stream);

  output_archive(*this);
}

static std::shared_ptr<Model> load(const std::string &file_name) {
  std::ifstream file_stream = fortis::handle_ifstream(file_name, std::ios::binary);

  cereal::BinaryInputArchive input_archive(file_stream);
  std::shared_ptr<Model> deserialized_model(new Model());
  input_archive(*deserialized_model);

  return deserialized_model;
}

} // namespace fortis