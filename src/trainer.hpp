#pragma once

#include <cereal/access.hpp>
#include <memory>
#include <src/model.hpp>
#include <stdexcept>

namespace fortis::trainers {

class GradientDescentTrainer {
public:
  GradientDescentTrainer(std::shared_ptr<Model> model, float learning_rate);

  void takeDescentStep();
  std::shared_ptr<Model> getModel() const;

private:
  void
  updateWeightMatrixParameter(std::vector<std::vector<float>> &weight_matrix,
                              std::vector<std::vector<float>> &jacobian);
  void updateBiasVectorParameter(std::vector<float> &bias_vector,
                                 std::vector<float> &gradient);

  std::shared_ptr<Model> _model;
  float _learning_rate;

  friend class cereal::access;
  template <typename Archive> void serialize(Archive &archive) {
    archive(_model, _learning_rate);
  }
};

} // namespace fortis::trainers