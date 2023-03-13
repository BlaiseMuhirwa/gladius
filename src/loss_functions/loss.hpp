
#include <_types/_uint32_t.h>
#include <src/cereal/access.hpp>
#include <vector>

namespace fortis::loss_functions {

static float softmax(float current_logit, const std::vector<float> &input) {
  float sum = 0.0f;
  std::for_each(input.begin(), input.end(),
                [&sum](float logit) { sum += exp(logit); });
  return exp(current_logit) / sum;
}

struct Loss {
  ~Loss() = default;

  virtual float getValue() = 0;
};

/**
 * Here we use the cross-entropy loss always with the softmax activation
 * For more about the cross-entropy loss with the Softmax activation
 * check out this page:
 * https://d2l.ai/chapter_linear-classification/softmax-regression.html#the-softmax
 *
 */
struct CrossEntropyLoss : public Loss {
  CrossEntropyLoss(const std::vector<float> &logits,
                   const std::vector<uint32_t> &label)
      : _logits(std::move(logits)), _label(std::move(label)) {}

  float getValue() final {
    assert(!_logits.empty());
    assert(!_label.empty());
    assert(_logits.size() == _label.size());

    float log_sum_exp = 0.0f;
    std::for_each(_logits.begin(), _logits.end(),
                  [&log_sum_exp](float logit) { log_sum_exp += exp(logit); });
    log_sum_exp = log(log_sum_exp);
    float second_term = 0.0f;
    for (uint32_t logit_index = 0; logit_index < _logits.size();
         logit_index++) {
      second_term += (_logits[logit_index] * _label[logit_index]);
    }
    float loss = log_sum_exp - second_term;
    return loss;
  }

  std::vector<float> _logits;
  std::vector<uint32_t> _label;
  float _value;

  template <typename Archive> void serialize(Archive &archive) {
    archive(_logits, _label, _value);
  }
};

} // namespace fortis::loss_functions