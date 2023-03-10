
#include "model.hpp"
#include <cereal/access.hpp>
#include <memory>

namespace fortis::trainers {

class StochasticGradientDescentTrainer {
public:
  StochasticGradientDescentTrainer(std::shared_ptr<Model> &model,
                                   float learning_rate)
      : _model(std::move(model)), _learning_rate(learning_rate) {}

  void update();
  std::shared_ptr<Model> getModel() const { return _model; }

private:
  std::shared_ptr<Model> _model;
  float _learning_rate;

  friend class cereal::access;
  template <typename Archive> void serialize(Archive &archive) {
    archive(_model, _learning_rate);
  }
};

} // namespace fortis::trainers