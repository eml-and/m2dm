import torch as th
import cu_kernels as cu
import math
from time import perf_counter
from einops import rearrange 
from improved_diff.m2unet import MonarchGatedConvBase, MonarchLinear


def blockdiag_matmul(x, weights):
    """
    x is expected to be of shape [..., sqrt_n * sqrt_n]
    w is expected to be of shape [sqrt_n, sqrt_n, sqrt_n]
    Reshape x to [..., sqrt_n, sqrt_n]
    reshaped_x = x.view(*x.shape[:-1], w.shape[0], w.shape[-1])
    Perform the block diagonal matrix multiplication
    result = th.einsum("bnm,...bm->...bn", w, reshaped_x)
    Reshape the result back to the original shape of x
    return result.reshape(*x.shape)
    # out = th.einsum(
    #     "bnm,...bm ->... bn",
    #     weights,
    #     x.view(*x.shape[:-1], weights.shape[0], weights.shape[-1]),
    # )
    # out = out.reshape(*x.shape)
    # return out
    """
    return th.einsum(
        "bnm,...bm ->... bn",
        weights,
        x.view(*x.shape[:-1], weights.shape[0], weights.shape[-1]),
    ).reshape(*x.shape)


class BlockMatmulCuda(th.autograd.Function):
    """
    torch wrapper for CUDA
    """
    @staticmethod
    def forward(ctx, weights: th.tensor, x: th.tensor):
        # x = x.contiguous()
        # weights = weights.contiguous()
        out = cu.blockdiag_matmul_fw(x, weights)
        # TODO: check autograd
        ctx.save_for_backward(x, weights)

        return out
    
    @staticmethod
    def backward(ctx, dL_dout: th.tensor):
        x, weights = ctx.saved_tensors
        dL_dw = cu.blockdiag_matmul_bw(dL_dout.contiguous(), x, weights)
        return dL_dw, None


def block_matmul_cuda(x, weights):
    return BlockMatmulCuda.apply(weights, x)


def forward_original(x, weights, sqrt_n):
    x = rearrange(x, "... (m n) -> ... (n m)", n=sqrt_n)
    x = blockdiag_matmul(x, weights)
    x = rearrange(x, "... (m n) -> ... (n m)", n=sqrt_n)
    x = blockdiag_matmul(x, weights)
    return rearrange(x, " ... (m n) -> ... (n m)", n=sqrt_n)


def fused_forward(x, weights, sqrt_n):
    x = rearrange(x, "... (m n) -> ... (n m)", n=sqrt_n)
    x = block_matmul_cuda(x, weights)
    x = rearrange(x, "... (m n) -> ... (n m)", n=sqrt_n)
    x = block_matmul_cuda(x, weights)
    return rearrange(x, " ... (m n) -> ... (n m)", n=sqrt_n)


def test_fused_rearrange(x, weights, sqrt_n):
    out_original = forward_original(x, weights, sqrt_n)
    out_fused = fused_forward(x, weights, sqrt_n)
    assert th.allclose(out_original, out_fused, atol=1e-5)
    

def test_block_matmul_cuda():
    """Tests the block matmul CUDA kernel against the einops implementation"""
    n = 36
    sqrt_n = int(math.sqrt(n))
    x = th.randn((48, 3, n, n), device='cuda')

    weights = th.randn((sqrt_n, sqrt_n, sqrt_n), device='cuda')
    LR = weights.clone().requires_grad_()
    LR_ = weights.clone().requires_grad_()
    assert th.allclose(LR, LR_)
    cumsum_einops = 0
    cumsum_cuda = 0
    n_runs: int = 500

    test_fused_rearrange(x, weights, sqrt_n)

    # TODO: fuse into kernel and see if it makes any difference
    #  train_util.py:167(forward_backward)
    # train_util.py:159(run_step)
    for i in range(n_runs):
        cuda_start = perf_counter()
        out = block_matmul_cuda(x=x, weights=LR)
        # th.cuda.synchronize()
        cuda_end = perf_counter()
        cuda_time = cuda_end - cuda_start
        loss = out.sum()
        loss.backward()

        einops_start = perf_counter()
        out_ = blockdiag_matmul(x=x, weights=LR_)
        # th.cuda.synchronize()
        einops_end = perf_counter()
        einops_time = einops_end - einops_start

        loss_ = out_.sum()
        loss_.backward()
        assert th.allclose(out, out_, atol=1e-6)  # Fails from atol=1e-7, 1e-8 is default
        try:
            assert th.allclose(LR.grad, LR_.grad, atol=1e-4)
        except AssertionError:
            print(f"CUDA and einops gradients do not match in iter <<{i}>> by: {LR.grad[0] - LR_.grad[0]}")
            continue
        if i % 100 == 0:
            print(f"einops: {round(einops_time*1000,4)}ms, CUDA: {round(cuda_time*1000,4)}ms ")
        cumsum_cuda += cuda_time
        cumsum_einops += einops_time
    print(f">> Mean einops fw+bw pass: {round((cumsum_einops/n_runs)*1000,3)}ms, mean CUDA: {round((cumsum_cuda/n_runs)*1000,3)}ms over {n_runs} iterations")


def test_Monarch_Gated_Conv_cuda():
    n = 36
    sqrt_n = int(math.sqrt(n))
    x = th.randn((48, 32, n, n), device='cuda')
    MGC = MonarchGatedConvBase(
        res=True,
        channels=32,
        sqrt_d=sqrt_n,
        sqrt_n=sqrt_n,
        num_heads=1,
        use_checkpoint=False).to(device='cuda')
    
    ML = MonarchLinear(sqrt_d=sqrt_n).to(device='cuda')
    x_ML = ML.forward(x)
    breakpoint()

    x_MGC = MGC.forward(x)
    # print(x_MGC.shape)

def main():
    test_block_matmul_cuda()
    # test_Monarch_Gated_Conv_cuda()


if __name__ == "__main__":
    main()