
#include "CUDA_advect_testkernel.h"

#include "CUDA_GLE_Solver.cuh"
#include "Particle.cuh"

#include "util/VectorMath_CUDA.cuh"

__device__ void advect(particle *p, float par_dt)
{
  vec3 dist{ (p->velMean._1 + p->velFluct._1) * par_dt,
             (p->velMean._2 + p->velFluct._2) * par_dt,
             (p->velMean._3 + p->velFluct._3) * par_dt };

  p->pos._1 = p->pos._1 + dist._1;
  p->pos._2 = p->pos._2 + dist._2;
  p->pos._3 = p->pos._3 + dist._3;

  p->delta_velFluct._1 = p->velFluct._1 - p->velFluct_old._1;
  p->delta_velFluct._2 = p->velFluct._2 - p->velFluct_old._2;
  p->delta_velFluct._3 = p->velFluct._3 - p->velFluct_old._3;

  p->velFluct_old = p->velFluct;
}

__global__ void testCUDA_advection(int length, particle *d_particle_list)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int it = index; it < length; it += stride) {
    solve(&d_particle_list[it], 1.0f, 0.0000001f, 10.0f);
    advect(&d_particle_list[it], 1.0f);
  }
  return;
}


void test_gpu(const int &length)
{

  int gpuID = 0;
  cudaError_t errorCheck = cudaGetDevice(&gpuID);

  int blockCount = 1;
  cudaDeviceGetAttribute(&blockCount, cudaDevAttrMultiProcessorCount, gpuID);
  std::cout << blockCount << std::endl;

  int threadsPerBlock = 32;
  cudaDeviceGetAttribute(&threadsPerBlock, cudaDevAttrMaxThreadsPerBlock, gpuID);
  std::cout << threadsPerBlock << std::endl;

  int blockSize = 1024;
  dim3 numberOfThreadsPerBlock(blockSize, 1, 1);
  dim3 numberOfBlocks(ceil(length / (float)(blockSize)), 1, 1);

  particle tmp;

  tmp.isRogue = false;
  tmp.isActive = true;
  tmp.CoEps = 0.0f;
  tmp.pos = { 1.0f, 1.0f, 1.0f };
  tmp.velMean = { 1.0f, 0.0f, 0.0f };
  tmp.velFluct = { 0.0f, 0.0f, 0.0f };

  std::vector<particle> particle_list;
  particle_list.resize(length, tmp);

  std::cout << "--------------------------------------" << std::endl;
  std::cout << "Particle 0" << std::endl;
  std::cout << " position: " << particle_list[0].pos._1 << ", " << particle_list[0].pos._2 << ", " << particle_list[0].pos._3 << std::endl;
  std::cout << " velocity: " << particle_list[0].velMean._1 << ", " << particle_list[0].velMean._2 << ", " << particle_list[0].velMean._3 << std::endl;
  std::cout << " fluct   : " << particle_list[0].velFluct._1 << ", " << particle_list[0].velFluct._2 << ", " << particle_list[0].velFluct._3 << std::endl;
  std::cout << "--------------------------------------" << std::endl;

  if (errorCheck == cudaSuccess) {
    // temp

    particle *d_particle_list;
    cudaMalloc((void **)&d_particle_list, length * sizeof(particle));


    auto gpuStartTime = std::chrono::high_resolution_clock::now();

    // copy to the device
    cudaMemcpy(d_particle_list, particle_list.data(), length * sizeof(particle), cudaMemcpyHostToDevice);

    // call kernel
    auto kernelStartTime = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < 1E5; ++k) {
      testCUDA_advection<<<numberOfBlocks, numberOfThreadsPerBlock>>>(length, d_particle_list);
      cudaDeviceSynchronize();
    }
    auto kernelEndTime = std::chrono::high_resolution_clock::now();

    // cudamemcpy back to host
    cudaMemcpy(particle_list.data(), d_particle_list, length * sizeof(particle), cudaMemcpyDeviceToHost);

    auto gpuEndTime = std::chrono::high_resolution_clock::now();

    // cudafree
    cudaFree(d_particle_list);

    std::chrono::duration<double> kernelElapsed = kernelEndTime - kernelStartTime;
    std::cout << "kernel  elapsed time: " << kernelElapsed.count() << " s\n";
    std::chrono::duration<double> gpuElapsed = gpuEndTime - gpuStartTime;
    std::cout << "GPU  elapsed time: " << gpuElapsed.count() << " s\n";

    std::cout << "--------------------------------------" << std::endl;
    std::cout << "Particle 0" << std::endl;
    std::cout << " position: " << particle_list[0].pos._1 << ", " << particle_list[0].pos._2 << ", " << particle_list[0].pos._3 << std::endl;
    std::cout << " velocity: " << particle_list[0].velMean._1 << ", " << particle_list[0].velMean._2 << ", " << particle_list[0].velMean._3 << std::endl;
    std::cout << " fluct   : " << particle_list[0].velFluct._1 << ", " << particle_list[0].velFluct._2 << ", " << particle_list[0].velFluct._3 << std::endl;
    std::cout << "--------------------------------------" << std::endl;


  } else {
    printf("CUDA ERROR!\n");
  }
}
