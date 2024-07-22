#include <torch/extension.h>
#include "utils.h"


torch::Tensor blockdiag_matmul(
    torch::Tensor x,
    torch::Tensor LR
){
    CHECK_INPUT(x);
    CHECK_INPUT(LR);

    return blockdiag_matmul_fw_cu(x, LR);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("blockdiag_matmul", &blockdiag_matmul);
}
