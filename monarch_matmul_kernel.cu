// #include <cuda_runtime.h>
// #include <torch/extension.h>
// #include <iostream>

// #include <cuda_runtime.h>
// #include <torch/extension.h>
// #include <iostream>

// // FORWARD
// __global__ void monarch_matmul_kernel(
//     const torch::PackedTensorAccessor<float, 5, torch::RestrictPtrTraits> x,
//     const float* __restrict__ monarch_matrix,
//     float* __restrict__ output,
//     int batch_size,
//     int channels,
//     int sqrt_n,
//     int sqrt_d) {

//     // Calculate indices
//     int batch_idx = blockIdx.x;
//     int channel_idx = blockIdx.y;
//     int row = threadIdx.y;
//     int col = threadIdx.x;

//     if (batch_idx < batch_size && channel_idx < channels && row < sqrt_n && col < sqrt_n) {
//         float result = 0.0f;

//         // Perform block matrix multiplication
//         for (int k = 0; k < sqrt_n; ++k) {
//             int input_idx = (batch_idx * channels * sqrt_n * sqrt_d) + 
//                             (channel_idx * sqrt_n * sqrt_d) +
//                             (row * sqrt_d) + k;
//             int matrix_idx = (row * sqrt_n + k) * sqrt_d + col;

//             result += input[input_idx] * monarch_matrix[matrix_idx];
//         }

//         int output_idx = (batch_idx * channels * sqrt_n * sqrt_d) + 
//                          (channel_idx * sqrt_n * sqrt_d) +
//                          (row * sqrt_d) + col;

//         output[output_idx] = result;
//     }
// }

// torch::Tensor monarch_matrix_mul(
//     const torch::Tensor input,
//     const torch::Tensor monarch_matrix) {

//     // Extract dimensions
//     const int batch_size = input.size(0);
//     const int channels = input.size(1);
//     const int sqrt_n = input.size(2);
//     const int sqrt_d = input.size(3);

//     // Create output tensor
//     torch::Tensor output = torch::empty_like(input);

//     // Kernel configuration
//     const dim3 threads(sqrt_n, sqrt_d);
//     const dim3 blocks(batch_size, channels);

//     // Launch kernel
//     monarch_matrix_mul_kernel<<<blocks, threads>>>(
//         input.data_ptr<float>(),
//         monarch_matrix.data_ptr<float>(),
//         output.data_ptr<float>(),
//         batch_size,
//         channels,
//         sqrt_n,
//         sqrt_d);

//     // Error checking
//     cudaError_t error = cudaGetLastError();
//     if (error != cudaSuccess) {
//         std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
//     }
//     cudaDeviceSynchronize();

//     return output;
// }
