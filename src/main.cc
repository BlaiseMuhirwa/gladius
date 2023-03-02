
#include "parameters.hpp"
#include <cassert>
#include <memory>

using fortis::parameters::Parameter;
using fortis::parameters::ParameterPointer;

int main(int argc, char **argv) {

  std::vector<std::vector<float>> input{};
  ParameterPointer parameter = std::make_shared<Parameter>(input);

  parameter->save("param.out");
  auto loaded_param = Parameter::load("param.out");

  assert(parameter->axes() == loaded_param->axes());

  return 0;
}