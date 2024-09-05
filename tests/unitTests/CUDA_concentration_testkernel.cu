
#include "CUDA_concentration_testkernel.h"

#include "util/VectorMath_CUDA.cuh"
#include "CUDA_particle_partition.cuh"

// #include "CUDA_GLE_Solver.cuh"
//  #include "Particle.cuh"

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

__device__ __managed__ ConcentrationParam param;

__global__ void set_positions(int length, particle_array d_particle_list, float *d_RNG_vals)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < length) {
    d_particle_list.pos[idx] = { 100.0f * d_RNG_vals[idx],
                                 100.0f * d_RNG_vals[idx + length],
                                 100.0f * d_RNG_vals[idx + 2 * length] };
  }
}


__global__ void collect(int length, particle_array d_particle_list, int *pBox, const ConcentrationParam param)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < length) {
    if (d_particle_list.state[idx] == ACTIVE) {
      // x-direction
      int i = floor((d_particle_list.pos[idx]._1 - param.lbndx) / (param.dx + 1e-9));
      // y-direction
      int j = floor((d_particle_list.pos[idx]._2 - param.lbndy) / (param.dy + 1e-9));
      // z-direction
      int k = floor((d_particle_list.pos[idx]._3 - param.lbndz) / (param.dz + 1e-9));

      if (i >= 0 && i <= param.nx - 1 && j >= 0 && j <= param.ny - 1 && k >= 0 && k <= param.nz - 1) {
        int id = k * param.ny * param.nx + j * param.nx + i;
        atomicAdd(&pBox[id], 1);
        // conc[id] = conc[id] + par.m * par.wdecay * timeStep;
      }
    }
  }
}

void test_gpu(const int &ntest, const int &new_particle, const int &length)
{

  int gpuID = 0;
  cudaError_t errorCheck = cudaGetDevice(&gpuID);

  if (errorCheck == cudaSuccess) {
    int blockCount = 1;
    cudaDeviceGetAttribute(&blockCount, cudaDevAttrMultiProcessorCount, gpuID);
    // std::cout << blockCount << std::endl;

    int threadsPerBlock = 128;
    cudaDeviceGetAttribute(&threadsPerBlock, cudaDevAttrMaxThreadsPerBlock, gpuID);
    // std::cout << threadsPerBlock << std::endl;

    auto gpuStartTime = std::chrono::high_resolution_clock::now();

    curandGenerator_t gen;
    float *d_RNG_vals;

    // Create pseudo-random number generator
    // CURAND_CALL(
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    // Allocate n floats on device to hold random numbers
    // Allocate numParticle * 3 floats on host
    int n = length * 3;
    // CUDA_CALL();
    cudaMalloc((void **)&d_RNG_vals, n * sizeof(float));

    // concnetration calculation
    param.lbndx = 0.0;
    param.lbndy = 0.0;
    param.lbndz = 0.0;

    param.ubndx = 100.0;
    param.ubndy = 100.0;
    param.ubndz = 100.0;

    param.nx = 20;
    param.ny = 20;
    param.nz = 20;

    param.dx = (param.ubndx - param.lbndx) / (param.nx);
    param.dy = (param.ubndy - param.lbndy) / (param.ny);
    param.dz = (param.ubndz - param.lbndz) / (param.nz);

    std::vector<int> h_pBox(param.nx * param.ny * param.nz, 0.0);

    int *d_pBox;
    cudaMalloc(&d_pBox, param.nx * param.ny * param.nz * sizeof(int));

    // Allocate particle array on the device ONLY
    particle_array d_particle_list;
    allocate_device_particle_list(d_particle_list, length);

    // initialize on the device
    cudaMemset(d_particle_list.state, INACTIVE, length * sizeof(int));
    cudaMemset(d_particle_list.ID, 0, length * sizeof(uint32_t));

    int blockSize = 256;
    int num_particle = length;// h_lower_count + new_particle;
    int numBlocks_all_particle = (num_particle + blockSize - 1) / blockSize;

    float ongoingAveragingTime = 0;
    float timeStep = 1;
    float volume = param.dx * param.dy * param.dz;

    // call kernel
    auto kernelStartTime = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < ntest; ++k) {

      cudaMemset(d_particle_list.state, ACTIVE, length * sizeof(int));
      curandGenerateUniform(gen, d_RNG_vals, 3 * length);
      cudaDeviceSynchronize();

      set_positions<<<numBlocks_all_particle, blockSize>>>(num_particle, d_particle_list, d_RNG_vals);
      cudaDeviceSynchronize();

      collect<<<numBlocks_all_particle, blockSize>>>(num_particle, d_particle_list, d_pBox, param);
      cudaDeviceSynchronize();
      ongoingAveragingTime += timeStep;
    }

    auto kernelEndTime = std::chrono::high_resolution_clock::now();

    // cudamemcpy back to host
    cudaMemcpy(h_pBox.data(), d_pBox, param.nx * param.ny * param.nz * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "ongoingAveragingTime = " << ongoingAveragingTime << std::endl;
    std::cout << "volume = " << volume << std::endl;

    /*
    for (auto &p : h_pBox) {
      p = p / (ongoingAveragingTime * volume);
    }
    */

    float m = 0, s = 0;
    for (auto &p : h_pBox) {
      m += p;
    }
    m /= (float)(param.nx * param.ny * param.nz);
    std::cout << "mean = " << m << std::endl;
    for (auto &p : h_pBox) {
      s += std::pow((float)(p - m), 2);
    }
    s = pow(s / (float)(param.nx * param.ny * param.nz), 0.5);
    std::cout << "std = " << s << std::endl;

    // cudafree
    free_device_particle_list(d_particle_list);

    cudaFree(d_pBox);
    cudaFree(d_RNG_vals);

    auto gpuEndTime = std::chrono::high_resolution_clock::now();


    std::cout << "--------------------------------------" << std::endl;
    std::chrono::duration<double> kernelElapsed = kernelEndTime - kernelStartTime;
    std::cout << "kernel elapsed time: " << kernelElapsed.count() << " s\n";
    std::chrono::duration<double> gpuElapsed = gpuEndTime - gpuStartTime;
    std::cout << "GPU elapsed time:    " << gpuElapsed.count() << " s\n";
  } else {
    printf("CUDA ERROR!\n");
  }
}
