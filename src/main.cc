
#include "parameters.h"
#include <memory>

using fortis::parameters::Parameter;
using fortis::parameters::ParameterPtr;

int main(int argc, char** argv) {

    std::vector<std::vector<float>> input{};
    ParameterPtr ptr = std::make_shared<Parameter>(input);

    return 0;
}