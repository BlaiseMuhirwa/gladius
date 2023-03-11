
#include <src/cereal/access.hpp>
#include <vector>


namespace fortis {

/**
 * For more about the cross-entropy loss with the Softmax activation
 * check out this page: 
 * https://d2l.ai/chapter_linear-classification/softmax-regression.html#the-softmax
 *
*/
struct CrossEntropyLoss {
    explicit CrossEntropyLoss(const std::vector<float>& logits): _logits(logits) {}

    void value() {
        return;
    }

    std::vector<float> _logits;
    float _value;
    
    template <typename Archive>
    void serialize(Archive& archive) {
        archive(_logits, _value);
    }
};

} // namespace