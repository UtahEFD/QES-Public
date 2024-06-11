#include "CUDA_vector_testkernel.h"
#include "util/VectorMath_CUDA.cuh"


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
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int it = index; it < length; it += stride) {
    multiply(d_A[it], d_b[it], d_x[it]);
  }
}

__global__ void testCUDA_invert(int length, mat3 *d_A)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int it = index; it < length; it += stride) {
    bool tt = invert(d_A[it]);
  }
}

__global__ void testCUDA_invariant(int length, mat3sym *d_tau, vec3 *d_invar)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int it = index; it < length; it += stride) {
    calcInvariants(d_tau[it], d_invar[it]);
  }
}

__global__ void testCUDA_realizable(int length, mat3sym *d_tau, vec3 *d_invar)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int it = index; it < length; it += stride) {
    makeRealizable(10e-4, d_tau[it]);
  }
}

void test_matrix_multiplication_gpu(const int &length, std::vector<mat3> &A, std::vector<vec3> &b, std::vector<vec3> &x)
{

  int gpuID = 0;
  cudaError_t errorCheck = cudaGetDevice(&gpuID);

  int blockCount = 1;
  cudaDeviceGetAttribute(&blockCount, cudaDevAttrMultiProcessorCount, gpuID);
  // std::cout << blockCount << std::endl;

  int threadsPerBlock = 32;
  cudaDeviceGetAttribute(&threadsPerBlock, cudaDevAttrMaxThreadsPerBlock, gpuID);
  // std::cout << threadsPerBlock << std::endl;

  int blockSize = 1024;
  dim3 numberOfThreadsPerBlock(blockSize, 1, 1);
  dim3 numberOfBlocks(ceil(length / (float)(blockSize)), 1, 1);

  if (errorCheck == cudaSuccess) {
    // temp

    mat3 *d_A;
    cudaMalloc((void **)&d_A, 9 * length * sizeof(float));
    vec3 *d_b;
    cudaMalloc((void **)&d_b, 3 * length * sizeof(float));
    vec3 *d_x;
    cudaMalloc((void **)&d_x, length * sizeof(vec3));

    auto gpuStartTime = std::chrono::high_resolution_clock::now();

    // copy to the device
    cudaMemcpy(d_A, A.data(), length * sizeof(mat3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), length * sizeof(vec3), cudaMemcpyHostToDevice);

    // call kernel
    auto kernelStartTime = std::chrono::high_resolution_clock::now();
    // testCUDA_vectormath<<<numberOfBlocks, numberOfThreadsPerBlock>>>();
    testCUDA_multiply<<<numberOfBlocks, numberOfThreadsPerBlock>>>(length, d_A, d_b, d_x);
    // testCUDA_invert<<<numberOfBlocks, numberOfThreadsPerBlock>>>(length, d_A);
    // testCUDA_invariant<<<numberOfBlocks, numberOfThreadsPerBlock>>>(length, d_tau, d_invar);
    //  testCUDA_advection<<<numberOfBlocks, numberOfThreadsPerBlock>>>(length, d_x);
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

  int blockCount = 1;
  cudaDeviceGetAttribute(&blockCount, cudaDevAttrMultiProcessorCount, gpuID);
  // std::cout << blockCount << std::endl;

  int threadsPerBlock = 32;
  cudaDeviceGetAttribute(&threadsPerBlock, cudaDevAttrMaxThreadsPerBlock, gpuID);
  // std::cout << threadsPerBlock << std::endl;

  int blockSize = 1024;
  dim3 numberOfThreadsPerBlock(blockSize, 1, 1);
  dim3 numberOfBlocks(ceil(length / (float)(blockSize)), 1, 1);

  if (errorCheck == cudaSuccess) {
    // temp

    mat3 *d_A;
    cudaMalloc((void **)&d_A, 9 * length * sizeof(float));

    auto gpuStartTime = std::chrono::high_resolution_clock::now();

    // copy to the device
    cudaMemcpy(d_A, A.data(), length * sizeof(mat3), cudaMemcpyHostToDevice);

    // call kernel
    auto kernelStartTime = std::chrono::high_resolution_clock::now();
    testCUDA_invert<<<numberOfBlocks, numberOfThreadsPerBlock>>>(length, d_A);
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

  int blockCount = 1;
  cudaDeviceGetAttribute(&blockCount, cudaDevAttrMultiProcessorCount, gpuID);
  // std::cout << blockCount << std::endl;

  int threadsPerBlock = 32;
  cudaDeviceGetAttribute(&threadsPerBlock, cudaDevAttrMaxThreadsPerBlock, gpuID);
  // std::cout << threadsPerBlock << std::endl;

  int blockSize = 1024;
  dim3 numberOfThreadsPerBlock(blockSize, 1, 1);
  dim3 numberOfBlocks(ceil(length / (float)(blockSize)), 1, 1);

  if (errorCheck == cudaSuccess) {
    // temp

    mat3sym *d_A;
    cudaMalloc((void **)&d_A, length * sizeof(mat3sym));
    vec3 *d_x;
    cudaMalloc((void **)&d_x, length * sizeof(vec3));

    auto gpuStartTime = std::chrono::high_resolution_clock::now();

    // copy to the device
    cudaMemcpy(d_A, A.data(), length * sizeof(mat3sym), cudaMemcpyHostToDevice);

    // call kernel
    auto kernelStartTime = std::chrono::high_resolution_clock::now();
    testCUDA_invariant<<<numberOfBlocks, numberOfThreadsPerBlock>>>(length, d_A, d_x);
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
