#include "model.hpp"
#include <random>
namespace fortis {

static inline float const MEAN = 0.0f;
static inline float const STD_DEV = 1.0f;

Parameter& addParameter(uint32_t dimension) {
    std::random_device random_device;
    std::mt19937 generator(random_device());
    std::normal_distribution<float> distribution(MEAN, STD_DEV);


}

} // namespace fortis