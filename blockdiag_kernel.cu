#include <cuda_runtime.h>
#include <torch/extension.h>
#include <iostream>


// FORWARD
// __global__ void blockdiag_matmul_fw_kernel(
//     const torch::PackedTensorAccessor<float, 5, torch::RestrictPtrTraits> x,
//     const torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits> weights,
//     torch::PackedTensorAccessor<float, 5, torch::RestrictPtrTraits> out,
//     int batch_size,
//     int channels, 
//     int n,
//     int sqrt_n) {

//     // Calculate the global indices
//     const int batch_idx = blockIdx.x; // Batch index
//     const int channel_idx = blockIdx.y; // Channel index
//     const int block_idx = blockIdx.z; // Block index (in the n dimension)

//     // Calculate the index within the block
//     const int row = threadIdx.y;
//     const int col = threadIdx.x;

//     if (batch_idx < batch_size && channel_idx < channels && block_idx < n && row < sqrt_n && col < sqrt_n) {
//         // bnk,...bk -> bn
//         float value = 0.0f;
//         for (int k = 0; k < sqrt_n; ++k) {
//             value += weights[row][col][k] * x[batch_idx][channel_idx][block_idx][row][k];
//         }

//         out[batch_idx][channel_idx][block_idx][row][col] = value;
//     }
// }
__global__ void blockdiag_matmul_fw_kernel(
    const torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits> x, // 4D input tensor
    const torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits> weights,
    torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits> out, // Now 4D output tensor
    int batch_size,
    int channels, 
    int n,
    int sqrt_n) {

    // Calculate the global indices
    const int batch_idx = blockIdx.x;     // Batch index
    const int channel_idx = blockIdx.y;   // Channel index
    const int block_idx = blockIdx.z;     // Block index (in the n dimension)

    // Calculate the index within the block
    const int row = threadIdx.y;
    const int col = threadIdx.x;

    if (batch_idx < batch_size && channel_idx < channels && block_idx < n && row < sqrt_n && col < sqrt_n) {
        // Compute the flat index within the 4D input tensor
        int flat_idx = row * sqrt_n + col;

        // Access the appropriate element from the flattened input
        float value = 0.0f;
        for (int k = 0; k < sqrt_n; ++k) {
            // Compute the flat index for the k-th element in the input
            int input_flat_idx = row * sqrt_n + k;

            // Multiply with the corresponding weight and accumulate the result
            value += weights[row][col][k] * x[batch_idx][channel_idx][block_idx][input_flat_idx];
        }

        // Flatten the row and col indices into a single index for the output
        int output_flat_idx = row * sqrt_n + col;

        // Store the result in the 4D output tensor
        out[batch_idx][channel_idx][block_idx][output_flat_idx] = value;
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

    torch::Tensor out = torch::empty_like(x);

    // Define the grid and block dimensions
    const dim3 threads(sqrt_n, sqrt_n);  // One thread per element in the sqrt_n x sqrt_n block
    const dim3 blocks(batch_size, channels, n);  // Grid for batches, channels, and blocks

    blockdiag_matmul_fw_kernel<<<blocks, threads>>>(
    x.packed_accessor<float, 4, torch::RestrictPtrTraits>(),
    weights.packed_accessor<float, 3, torch::RestrictPtrTraits>(),
    out.packed_accessor<float, 4, torch::RestrictPtrTraits>(),
    batch_size,
    channels,
    n,
    sqrt_n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return out;
}


// BACKWARD
__global__ void blockdiag_matmul_bw_kernel(
    const torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits> dL_dout,
    const torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits> dL_dw, // same shape as weights
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
        // gradient computation
        for (int k = 0; k < sqrt_n; ++k) {
            atomicAdd(&dL_dw[row][col][k],
                      dL_dout[batch_idx][channel_idx][block_idx][row][col] * x[batch_idx][channel_idx][block_idx][row][k]);
        }
    }
}

torch::Tensor blockdiag_matmul_bw_cu(
    const torch::Tensor dL_dout,
    const torch::Tensor x,
    const torch::Tensor weights) {

    // Extract dimensions
    const int batch_size = x.size(0);
    const int channels = x.size(1);
    const int n = x.size(2); // Number of blocks
    const int sqrt_n = weights.size(0);

    // output tensor
    torch::Tensor dL_dw = torch::zeros_like(weights);

    // Define the grid and block dimensions
    const dim3 threads(sqrt_n, sqrt_n);  // One thread per element in the sqrt_n x sqrt_n block
    const dim3 blocks(batch_size, channels, n);  // Grid for batches, channels, and blocks

    blockdiag_matmul_bw_kernel<<<blocks, threads>>>(
    dL_dout.packed_accessor<float, 4, torch::RestrictPtrTraits>(),
    x.packed_accessor<float, 4, torch::RestrictPtrTraits>(),
    dL_dw.packed_accessor<float, 3, torch::RestrictPtrTraits>(),
    batch_size,
    channels,
    n,
    sqrt_n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return dL_dw;
}
