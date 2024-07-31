#include <torch/extension.h>
#include "utils.h"


torch::Tensor blockdiag_matmul_fw(
    const torch::Tensor x,
    const torch::Tensor LR
){
    CHECK_INPUT(x);
    CHECK_INPUT(LR);

    return blockdiag_matmul_fw_cu(x, LR);
}

torch::Tensor blockdiag_matmul_bw(
    const torch::Tensor dL_out,
    const torch::Tensor x,
    const torch::Tensor LR
){
    CHECK_INPUT(dL_out);
    CHECK_INPUT(x);
    CHECK_INPUT(LR);

    return blockdiag_matmul_bw_cu(dL_out, x, LR);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("blockdiag_matmul_fw", &blockdiag_matmul_fw);
     m.def("blockdiag_matmul_bw", &blockdiag_matmul_bw);
}
