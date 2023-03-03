
#include "parameters.hpp"
#include "src/comp_graph/vertex.hpp"
#include "src/operations/base_op.hpp"
#include <cassert>
#include <memory>
#include <src/parameters.hpp>

using fortis::Vertex;
using fortis::operations::Operation;
using fortis::parameters::Parameter;
using fortis::parameters::ParameterPointer;

int main(int argc, char **argv) {

  std::vector<std::vector<float>> input{};
  ParameterPointer parameter = std::make_shared<Parameter>(input);

  return 0;
}