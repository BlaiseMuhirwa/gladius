#pragma once

#include <cereal/access.hpp>
#include <src/model.hpp>
#include <memory>
#include <stdexcept>

namespace gladius::trainers {

class GradientDescentTrainer {
 public:
  GradientDescentTrainer(std::shared_ptr<Model> model, float learning_rate);

  void takeDescentStep();
  std::shared_ptr<Model> getModel();

  void zeroOutGradients();

 private:
  std::shared_ptr<Model> _model;
  float _learning_rate;

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(_model, _learning_rate);
  }
};

}  // namespace gladius::trainers