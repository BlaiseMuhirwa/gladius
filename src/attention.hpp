#pragma once

#include <cstdint>

/**
This class implements the scaled dot-product attention layer as
described in the original paper here:
https://arxiv.org/pdf/1706.03762.pdf
**/

namespace gladius::transformer {

class MultiHeadAttention {
  explicit MultiHeadAttention(uint32_t attention_blocks);
};

}  // namespace gladius::transformer