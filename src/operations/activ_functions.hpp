
#include "base_op.hpp"
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/access.hpp>
#include <memory>

namespace fortis {
using fortis::Vertex;
using fortis::Expression;

class TanHActivation : public Vertex,
                       public std::enable_shared_from_this<TanHActivation> {
public:
    void forward() final {
        void;
    }

    void backward() final {
        (void);
    }

private:
    std::vector<std::shared_ptr<Expression>> _incoming_edges;

};

CEREAL_REGISTER_TYPE(fortis::Vertex);


} // namespace fortis