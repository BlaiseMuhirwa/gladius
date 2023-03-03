
#include "model.hpp"
#include <cereal/access.hpp>
#include <memory>

namespace fortis {

class SGDTrainer {
  explicit SGDTrainer(std::shared_ptr<Model> &model)
      : _model(std::move(model)) {}

  void update();

private:
  std::shared_ptr<Model> _model;

  friend class cereal::access;
};

} // namespace fortis