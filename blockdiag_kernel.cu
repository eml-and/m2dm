#include <cuda_runtime.h>
#include <torch/extension.h>
#include <iostream>

// __global__ void blockdiag_matmul_fw_kernel(
//     const float* __restrict__ x,
//     const float* __restrict__ weights,
//     float* __restrict__ out,
//     int batch_size,
//     int channels,
//     int sqrt_n,
//     int n) {

//     const int batch_idx = blockIdx.x;
//     const int channel_idx = blockIdx.y;

//     int row = threadIdx.y;
//     int col = threadIdx.x;

//     if (batch_idx < batch_size && channel_idx < channels && row < sqrt_n && col < sqrt_n) {
//         float value = 0.0f;

//         for (int k = 0; k < sqrt_n; ++k) {
//             int x_idx = (batch_idx * channels * n * n) + 
//                         (channel_idx * n * n) + 
//                         (row * sqrt_n * sqrt_n) + 
//                         (k * sqrt_n) + 
//                         col;

//             int w_idx = (row * sqrt_n + col) * sqrt_n + k;

//             value += weights[w_idx] * x[x_idx];
//         }

//         int out_idx = (batch_idx * channels * n * n) + 
//                       (channel_idx * n * n) + 
//                       (row * sqrt_n) + 
//                       col;

//         out[out_idx] = value;
//     }
//     }


// torch::Tensor blockdiag_matmul_fw_cu(
//     const torch::Tensor x,
//     const torch::Tensor weights) {

//     const int batch_size = x.size(0);
//     const int channels = x.size(1);
//     const int n = x.size(-1);
//     const int sqrt_n = weights.size(0);

//     torch::Tensor out = torch::empty_like(x);

//     const dim3 threads(sqrt_n, sqrt_n);
//     const dim3 blocks(batch_size, channels);

//     blockdiag_matmul_fw_kernel<<<blocks, threads>>>(
//         x.data_ptr<float>(),
//         weights.data_ptr<float>(),
//         out.data_ptr<float>(),
//         batch_size,
//         channels,
//         sqrt_n,
//         n);

//     cudaError_t error = cudaGetLastError();
//     if (error != cudaSuccess) {
//         std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
//     }
//     cudaDeviceSynchronize();

//     return out;
// }

// CUDA kernel for block diagonal matrix multiplication
__global__ void blockdiag_matmul_fw_kernel(
    torch::PackedTensorAccessor<float, 5, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits> weights,
    torch::PackedTensorAccessor<float, 5, torch::RestrictPtrTraits> out,
    int batch_size,
    int channels, 
    int n,
    int sqrt_n) {

    // Calculate the global indices
    const int batch_idx = blockIdx.x; // Batch index
    const int channel_idx = blockIdx.y; // Channel index
    const int block_idx = blockIdx.z; // Block index (in the n dimension)

    // Calculate the index within the block
    const int row = threadIdx.y;
    const int col = threadIdx.x;

    if (batch_idx < batch_size && channel_idx < channels && block_idx < n && row < sqrt_n && col < sqrt_n) {
        // bnk,...bk -> bn
        float value = 0.0f;
        for (int k = 0; k < sqrt_n; ++k) {
            value += weights[row][col][k] * x[batch_idx][channel_idx][block_idx][row][k];
        }

        out[batch_idx][channel_idx][block_idx][row][col] = value;
    }
}

torch::Tensor blockdiag_matmul_fw_cu(
    const torch::Tensor x,
    const torch::Tensor weights) {

    // Extract dimensions
    const int batch_size = x.size(0);
    const int channels = x.size(1);
    const int n = x.size(2); // Number of blocks
    const int sqrt_n = weights.size(0);

    // Create an output tensor
    torch::Tensor out = torch::empty_like(x);

    // Define the grid and block dimensions
    const dim3 threads(sqrt_n, sqrt_n);  // One thread per element in the sqrt_n x sqrt_n block
    const dim3 blocks(batch_size, channels, n);  // Grid for batches, channels, and blocks

    // Launch the kernel
    blockdiag_matmul_fw_kernel<<<blocks, threads>>>(
    x.packed_accessor<float, 5, torch::RestrictPtrTraits>(),
    weights.packed_accessor<float, 3, torch::RestrictPtrTraits>(),
    out.packed_accessor<float, 5, torch::RestrictPtrTraits>(),
    batch_size,
    channels,
    n,
    sqrt_n);

    // Check for any errors during kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return out;
}