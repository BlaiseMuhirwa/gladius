#pragma once

#include <_types/_uint32_t.h>
#include <algorithm>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <memory>
#include <optional>
#include <src/cereal/access.hpp>
#include <src/comp_graph/vertices/activ_functions.hpp>
#include <src/comp_graph/vertices/vertex.hpp>
#include <stdexcept>
#include <vector>

namespace fortis::comp_graph {

using fortis::comp_graph::SoftMaxActivation;
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

  /*
   * The constructor expects input vertex to have an output with
   * the same dimension as the label. The output vector computed
   * by the input vertex consists of logits prior to a softmax
   * operation.
   */
  CrossEntropyLoss(VertexPointer input_vertex, std::vector<float> &label)
      : _input(std::move(input_vertex)), _label(std::move(label)) {
    if (_input->getOutputSize() != _label.size()) {
      throw std::invalid_argument(
          "The size of the logits vector must be equal to the size of the "
          "label vector. Logits have size " +
          std::to_string(_logits.size()) + " while the label vector has size " +
          std::to_string(_label.size()));
    }
    // This has to be a copy because std::move would cause a
    // runtime error for subsequent calls that require _input->getOutput()
    // to be non-empty
    _logits = _input->getOutput().at(0);
  }

  void forward() final {
    assert(!_label.empty());
    assert(!_logits.empty());
    assert(!_loss.has_value());

    applyOperation();
  }

  /**
   * Suppose P \in \mathbb{R}^k is the output probability vector computed
   * by the softmax function, and let CE(Y, P) be the cross entropy loss
   * computed by this vertex. Treating CE as a function of P (with Y) constant,
   * we observe that the gradient is a 1 x n matrix 
   */
  void backward(const std::optional<std::vector<std::vector<float>>> &gradient =
                    std::nullopt) final {
    if (gradient.has_value()) {
      throw std::invalid_argument("The loss function's backward method should "
                                  "not have a gradient parameter.");
    }
    assert(_gradient.empty());
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
    _gradient = std::vector<float>(_logits.size());

    for (uint32_t logit_index = 0; logit_index < _logits.size();
         logit_index++) {
      auto derivative =
          softmax(_logits[logit_index], _logits) - _label[logit_index];
      _gradient[logit_index] = derivative;
    }
  }

  inline std::string getName() final { return "CrossEntropyLoss"; }

  inline std::vector<std::vector<float>> getOutput() const override {
    assert(_loss.has_value());
    return {{_loss.value()}};
  }

  constexpr uint32_t getOutputSize() const final { return 1; }

private:
  /**
   * Assuming a one-hot encoded vector as an input, this function returns
   * the index in the vector where the label is 1.0
   */
  static uint32_t findIndexWithPositiveLabel(const std::vector<float> &label) {
    auto iterator = std::find(label.begin(), label.end(), 1.0);
    if (iterator != label.end()) {
      return iterator - label.begin();
    }
    throw std::runtime_error("Each label vector must be one-hot encoded.");
  }
  /**
   * Let Y and P be the true distribution of the labels and the computed
   * probabilities by the neural network respectively. Assuming that they
   * are supported on a space of n possible values, the cross entropy is
   * given by
   *        CE(Y, P) = \sum_{k=1}^{n}y_k \log(p_k)
   * The function below computes exactly the expression above.
   * For more on cross-entropy, check out
   * https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
   */
  std::shared_ptr<Vertex> applyOperation() final {
    _loss = 0.f;
    auto output_size = _label.size();
    auto probabilities = _input->getOutput().at(0);
    for (uint32_t prob_index = 0; prob_index < output_size; prob_index++) {
      (*_loss) += _label[prob_index] * log(probabilities[prob_index]);
    }
    return shared_from_this();
  }

  VertexPointer _input;
  std::vector<float> _logits;

  // One-hot encoded vector representing the label
  std::vector<float> _label;
  std::optional<float> _loss;

  template <typename Archive> void serialize(Archive &archive) {
    archive(cereal::base_class<Vertex>(this), _input, _label, _loss, _gradient);
  }
};

} // namespace fortis::comp_graph

CEREAL_REGISTER_TYPE(fortis::comp_graph::CrossEntropyLoss)