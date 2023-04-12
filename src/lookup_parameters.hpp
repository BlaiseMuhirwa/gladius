#pragma once

#include <cereal/access.hpp>
#include <cereal/types/vector.hpp>
#include <memory>
#include <optional>
#include <vector>
namespace fortis::parameters {

class LookupParameter {
 public:
  LookupParameter(uint32_t num_embeddings, uint32_t embedding_dim,
                  std::optional<uint32_t> padding_index = std::nullopt,
                  std::optional<uint32_t> max_norm = std::nullopt,
                  std::optional<std::string> norm_type = std::nullopt,
                  bool sparse = false)
      : _num_embeddings(num_embeddings),
        _embedding_dim(embedding_dim),
        _padding_index(padding_index),
        _max_norm(max_norm),
        _norm_type(std::move(norm_type)),
        _sparse(sparse){};

 private:
  uint32_t _num_embeddings;
  uint32_t _embedding_dim;
  std::optional<uint32_t> _padding_index;
  std::optional<uint32_t> _max_norm;
  std::optional<std::string> _norm_type;
  bool _sparse;

  LookupParameter() = default;

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(_num_embeddings, _embedding_dim, _padding_index, _max_norm,
            _norm_type, _sparse);
  }
};

}  // namespace fortis::parameters