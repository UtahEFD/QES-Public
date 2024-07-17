
#include "CUDA_buffer_testkernel.h"

#include "util/VectorMath_CUDA.cuh"

#include "CUDA_GLE_Solver.cuh"
#include "Particle.cuh"
#include "CUDA_buffer.cuh"


__global__ void test_kernel(ring *buffer, float *input_data_d, const int &nume_element)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  /*for (int tid = index; tid < length; tid += stride) {
    // PGD[tid].interp[tid].interpValues(TGD, p[tid].pos, p[tid].tau,p[tid].flux_div, p[tid].nuT, p[tid].CoEps);
    if (d_particle_list[tid].isActive && !d_particle_list[tid].isRogue) {

      d_particle_list[tid].tau = { 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f };
      d_particle_list[tid].fluxDiv = { 0.0f, 0.0f, 0.0f };

      solve(d_particle_list, tid, 1.0f, 0.0000001f, 10.0f);
      advect(d_particle_list, tid, 1.0f);
    }
    }*/
}

void test_gpu_buffer(const int &num_elements)
{
  int gpuID = 0;
  cudaError_t errorCheck = cudaGetDevice(&gpuID);

  int blockCount = 1;
  cudaDeviceGetAttribute(&blockCount, cudaDevAttrMultiProcessorCount, gpuID);
  std::cout << blockCount << std::endl;

  int threadsPerBlock = 128;
  cudaDeviceGetAttribute(&threadsPerBlock, cudaDevAttrMaxThreadsPerBlock, gpuID);
  std::cout << threadsPerBlock << std::endl;

  int blockSize = threadsPerBlock;
  dim3 numberOfThreadsPerBlock(blockSize, 1, 1);
  dim3 numberOfBlocks(ceil(num_elements / (float)(blockSize)), 1, 1);

  std::cout << numberOfThreadsPerBlock.x << " " << numberOfBlocks.x << std::endl;
  int num_threads = blockCount * threadsPerBlock;
  std::cout << num_threads << std::endl;

  if (errorCheck == cudaSuccess) {

    auto gpuStartTime = std::chrono::high_resolution_clock::now();

    ring buffer;
    float *input_data;

    buffer.N = 32;// Warp size
    buffer.capacity = 1024;// Power of 2
    cudaMallocHost(&buffer.head_h, sizeof(int) * num_threads);
    cudaMallocHost(&buffer.size_h, sizeof(int) * num_threads);
    cudaMallocHost(&buffer.peek_head_h, sizeof(size_t) * num_threads);
    cudaMallocHost(&buffer.data_h, sizeof(size_t) * buffer.capacity * buffer.N);
    // Initialize other data (input_data, etc.)
    // Allocate memory on the device
    ring *buffer_d;
    float *input_data_d;
    cudaMalloc(&buffer_d, sizeof(ring));
    cudaMalloc(&input_data_d, sizeof(float) * num_elements);
    // Copy data to the device
    cudaMemcpy(buffer_d, &buffer, sizeof(ring), cudaMemcpyHostToDevice);
    cudaMemcpy(input_data_d, input_data, sizeof(float) * num_elements, cudaMemcpyHostToDevice);
    // Launch the kernel
    test_kernel<<<numberOfBlocks, numberOfThreadsPerBlock>>>(buffer_d, input_data_d, num_elements);
    auto kernelStartTime = std::chrono::high_resolution_clock::now();
    // for (int k = 0; k < 1E4; ++k) {
    //   testCUDA_advection<<<numberOfBlocks, numberOfThreadsPerBlock>>>(num_elements, d_particle_list);
    // cudaDeviceSynchronize();
    // }
    auto kernelEndTime = std::chrono::high_resolution_clock::now();
    // Copy results back to the host (if needed)

    // Free memory
    cudaFree(buffer_d);
    cudaFree(input_data_d);
    cudaFreeHost(buffer.head_h);
    cudaFreeHost(buffer.size_h);
    cudaFreeHost(buffer.peek_head_h);
    cudaFreeHost(buffer.data_h);

    auto gpuEndTime = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> kernelElapsed = kernelEndTime - kernelStartTime;
    std::cout << "kernel  elapsed time: " << kernelElapsed.count() << " s\n";
    std::chrono::duration<double> gpuElapsed = gpuEndTime - gpuStartTime;
    std::cout << "GPU  elapsed time: " << gpuElapsed.count() << " s\n";

  } else {
    printf("CUDA ERROR!\n");
  }
}
