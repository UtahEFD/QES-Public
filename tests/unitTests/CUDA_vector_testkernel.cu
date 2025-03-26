#include "CUDA_vector_testkernel.h"
#include "util/VectorMath_CUDA.h"


__global__ void testCUDA_vectormath()
{
  // int id = (blockDim.x * blockIdx.x) + threadIdx.x;

  vec3 x, y, z;
  x = { 1.0f, 2.0f, 3.0f };
  y = { 1.0f, 2.0f, 3.0f };

  vec3 n = { 1.0f, 0.0f, 0.0f };

  z._1 = 3.0f * x._1 + y._1;
  z._2 = 3.0f * x._2 + y._2;
  z._3 = 3.0f * x._3 + y._3;

  float l = length(z);

  float s = dot(x, y);

  reflect(n, z);
}

__global__ void testCUDA_multiply(int length, mat3 *d_A, vec3 *d_b, vec3 *d_x)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    multiply(d_A[idx], d_b[idx], d_x[idx]);
  }
}

__global__ void testCUDA_invert(int length, mat3 *d_A)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    bool out = invert(d_A[idx]);
  }
}

__global__ void testCUDA_invariant(int length, mat3sym *d_tau, vec3 *d_invar)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    calcInvariants(d_tau[idx], d_invar[idx]);
  }
}

__global__ void testCUDA_realizable(int length, mat3sym *d_tau, vec3 *d_invar)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    makeRealizable(10e-4, d_tau[idx]);
  }
}

void test_matrix_multiplication_gpu(const int &length, std::vector<mat3> &A, std::vector<vec3> &b, std::vector<vec3> &x)
{

  int gpuID = 0;
  cudaError_t errorCheck = cudaGetDevice(&gpuID);

  if (errorCheck == cudaSuccess) {
    int blockSize = 256;
    int numBlocks = (length + blockSize - 1) / blockSize;

    mat3 *d_A;
    cudaMalloc((void **)&d_A, length * sizeof(mat3));
    vec3 *d_b;
    cudaMalloc((void **)&d_b, length * sizeof(vec3));
    vec3 *d_x;
    cudaMalloc((void **)&d_x, length * sizeof(vec3));

    auto gpuStartTime = std::chrono::high_resolution_clock::now();

    // copy to the device
    cudaMemcpy(d_A, A.data(), length * sizeof(mat3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), length * sizeof(vec3), cudaMemcpyHostToDevice);

    // call kernel
    auto kernelStartTime = std::chrono::high_resolution_clock::now();
    testCUDA_multiply<<<numBlocks, blockSize>>>(length, d_A, d_b, d_x);
    cudaDeviceSynchronize();
    auto kernelEndTime = std::chrono::high_resolution_clock::now();

    // cudamemcpy back to host
    // cudaMemcpy(A.data(), d_A, length * sizeof(mat3), cudaMemcpyDeviceToHost);
    cudaMemcpy(x.data(), d_x, length * sizeof(vec3), cudaMemcpyDeviceToHost);

    auto gpuEndTime = std::chrono::high_resolution_clock::now();

    // cudafree
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);

    std::chrono::duration<double> kernelElapsed = kernelEndTime - kernelStartTime;
    std::cout << "kernel  elapsed time: " << kernelElapsed.count() << " s\n";
    std::chrono::duration<double> gpuElapsed = gpuEndTime - gpuStartTime;
    std::cout << "GPU  elapsed time: " << gpuElapsed.count() << " s\n";

  } else {
    printf("CUDA ERROR!\n");
  }
}


void test_matrix_inversion_gpu(const int &length, std::vector<mat3> &A)
{

  int gpuID = 0;
  cudaError_t errorCheck = cudaGetDevice(&gpuID);

  if (errorCheck == cudaSuccess) {
    int blockSize = 256;
    int numBlocks = (length + blockSize - 1) / blockSize;

    mat3 *d_A;
    cudaMalloc((void **)&d_A, length * sizeof(mat3));

    auto gpuStartTime = std::chrono::high_resolution_clock::now();

    // copy to the device
    cudaMemcpy(d_A, A.data(), length * sizeof(mat3), cudaMemcpyHostToDevice);

    // call kernel
    auto kernelStartTime = std::chrono::high_resolution_clock::now();
    testCUDA_invert<<<numBlocks, blockSize>>>(length, d_A);
    cudaDeviceSynchronize();
    auto kernelEndTime = std::chrono::high_resolution_clock::now();

    // cudamemcpy back to host
    cudaMemcpy(A.data(), d_A, length * sizeof(mat3), cudaMemcpyDeviceToHost);

    auto gpuEndTime = std::chrono::high_resolution_clock::now();

    // cudafree
    cudaFree(d_A);

    std::chrono::duration<double> kernelElapsed = kernelEndTime - kernelStartTime;
    std::cout << "kernel  elapsed time: " << kernelElapsed.count() << " s\n";
    std::chrono::duration<double> gpuElapsed = gpuEndTime - gpuStartTime;
    std::cout << "GPU  elapsed time: " << gpuElapsed.count() << " s\n";

  } else {
    printf("CUDA ERROR!\n");
  }
}

void test_matrix_invariants_gpu(const int &length, std::vector<mat3sym> &A, std::vector<vec3> &x)
{

  int gpuID = 0;
  cudaError_t errorCheck = cudaGetDevice(&gpuID);
  if (errorCheck == cudaSuccess) {
    int blockSize = 256;
    int numBlocks = (length + blockSize - 1) / blockSize;

    mat3sym *d_A;
    cudaMalloc((void **)&d_A, length * sizeof(mat3sym));
    vec3 *d_x;
    cudaMalloc((void **)&d_x, length * sizeof(vec3));

    auto gpuStartTime = std::chrono::high_resolution_clock::now();

    // copy to the device
    cudaMemcpy(d_A, A.data(), length * sizeof(mat3sym), cudaMemcpyHostToDevice);

    // call kernel
    auto kernelStartTime = std::chrono::high_resolution_clock::now();
    testCUDA_invariant<<<numBlocks, blockSize>>>(length, d_A, d_x);
    cudaDeviceSynchronize();
    auto kernelEndTime = std::chrono::high_resolution_clock::now();

    // cudamemcpy back to host
    cudaMemcpy(x.data(), d_x, length * sizeof(vec3), cudaMemcpyDeviceToHost);

    auto gpuEndTime = std::chrono::high_resolution_clock::now();

    // cudafree
    cudaFree(d_A);
    cudaFree(d_x);

    std::chrono::duration<double> kernelElapsed = kernelEndTime - kernelStartTime;
    std::cout << "kernel  elapsed time: " << kernelElapsed.count() << " s\n";
    std::chrono::duration<double> gpuElapsed = gpuEndTime - gpuStartTime;
    std::cout << "GPU  elapsed time: " << gpuElapsed.count() << " s\n";

  } else {
    printf("CUDA ERROR!\n");
  }
}
