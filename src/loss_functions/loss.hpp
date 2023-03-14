
#include <_types/_uint32_t.h>
#include <memory>
#include <optional>
#include <src/cereal/access.hpp>
#include <src/comp_graph/vertex.hpp>
#include <stdexcept>
#include <vector>

namespace fortis::comp_graph {

using fortis::comp_graph::Vertex;
using fortis::comp_graph::VertexPointer;

static float softmax(float current_logit, const std::vector<float> &input) {
  float sum = 0.0f;
  std::for_each(input.begin(), input.end(),
                [&sum](float logit) { sum += exp(logit); });
  return exp(current_logit) / sum;
}

/**
 * Here we use the cross-entropy loss always with the softmax activation
 * For more about the cross-entropy loss with the Softmax activation
 * check out this page:
 * https://d2l.ai/chapter_linear-classification/softmax-regression.html#the-softmax
 *
 */
class CrossEntropyLoss final
    : public Vertex,
      public std::enable_shared_from_this<CrossEntropyLoss> {
  CrossEntropyLoss(const std::vector<float> &logits,
                   const std::vector<uint32_t> &label)
      : _logits(std::move(logits)), _label(std::move(label)) {}

  std::shared_ptr<Vertex>
  setIncomingEdges(std::vector<VertexPointer> &edges) final;

  void forward() final {
    assert(!_logits.empty());
    assert(!_label.empty());
    assert(_logits.size() == _label.size());

    applyOperation();
  }

  /**
   * Here we recall that the derivative of the cross-entropy loss function
   * with respect to any logit \hat{y_i} is given by
   * softmax(\hat{y_i}) - y_i
   * where y_i is the corresponding value in the label vector.
   */
  void backward(const std::optional<std::vector<std::vector<float>>> &gradient =
                    std::nullopt) final {
    if (gradient.has_value()) {
      throw std::invalid_argument("The loss function's backward method should "
                                  "not have a gradient parameter.");
    }
    assert(_gradient.empty());

    for (uint32_t logit_index = 0; logit_index < _logits.size();
         logit_index++) {
      auto derivative =
          softmax(_logits[logit_index], _logits) - _label[logit_index];
      _gradient.emplace_back(derivative);
    }
  }

  std::string getName() final { return "CrossEntropyLoss"; }

  std::vector<std::vector<float>> getOutput() const final { return {}; }

  constexpr float getValue() const {
    assert(_loss.has_value());
    return _loss.value();
  }

private:
  std::shared_ptr<Vertex> applyOperation() final {
    float log_sum_exp = 0.0f;
    std::for_each(_logits.begin(), _logits.end(),
                  [&log_sum_exp](float logit) { log_sum_exp += exp(logit); });
    log_sum_exp = log(log_sum_exp);
    float second_term = 0.0f;
    for (uint32_t logit_index = 0; logit_index < _logits.size();
         logit_index++) {
      second_term += (_logits[logit_index] * _label[logit_index]);
    }
    _loss = log_sum_exp - second_term;
    return shared_from_this();
  }

  std::vector<float> _logits;
  std::vector<uint32_t> _label;
  std::optional<float> _loss;
  std::vector<float> _gradient;

  template <typename Archive> void serialize(Archive &archive) {
    archive(_logits, _label, _loss, _gradient);
  }
};

} // namespace fortis::comp_graph