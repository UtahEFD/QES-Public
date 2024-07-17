
#include "CUDA_advect_testkernel.h"

#include "util/VectorMath_CUDA.cuh"

#include "CUDA_GLE_Solver.cuh"
#include "Particle.cuh"

__device__ void advect(particle_AOS *p, int tid, float par_dt)
{
  vec3 dist{ (p[tid].velMean._1 + p[tid].velFluct._1) * par_dt,
             (p[tid].velMean._2 + p[tid].velFluct._2) * par_dt,
             (p[tid].velMean._3 + p[tid].velFluct._3) * par_dt };

  p[tid].pos._1 = p[tid].pos._1 + dist._1;
  p[tid].pos._2 = p[tid].pos._2 + dist._2;
  p[tid].pos._3 = p[tid].pos._3 + dist._3;

  p[tid].delta_velFluct._1 = p[tid].velFluct._1 - p[tid].velFluct_old._1;
  p[tid].delta_velFluct._2 = p[tid].velFluct._2 - p[tid].velFluct_old._2;
  p[tid].delta_velFluct._3 = p[tid].velFluct._3 - p[tid].velFluct_old._3;

  p[tid].velFluct_old = p[tid].velFluct;
}

__device__ void advect(particle_SOA p, int tid, float par_dt)
{
  vec3 dist{ (p.velMean[tid]._1 + p.velFluct[tid]._1) * par_dt,
             (p.velMean[tid]._2 + p.velFluct[tid]._2) * par_dt,
             (p.velMean[tid]._3 + p.velFluct[tid]._3) * par_dt };

  p.pos[tid]._1 = p.pos[tid]._1 + dist._1;
  p.pos[tid]._2 = p.pos[tid]._2 + dist._2;
  p.pos[tid]._3 = p.pos[tid]._3 + dist._3;

  p.delta_velFluct[tid]._1 = p.velFluct[tid]._1 - p.velFluct_old[tid]._1;
  p.delta_velFluct[tid]._2 = p.velFluct[tid]._2 - p.velFluct_old[tid]._2;
  p.delta_velFluct[tid]._3 = p.velFluct[tid]._3 - p.velFluct_old[tid]._3;

  p.velFluct_old[tid] = p.velFluct[tid];
}

__global__ void testCUDA_advection(int length, particle_AOS *d_particle_list)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int tid = index; tid < length; tid += stride) {
    // PGD[tid].interp[tid].interpValues(TGD, p[tid].pos, p[tid].tau,p[tid].flux_div, p[tid].nuT, p[tid].CoEps);
    if (d_particle_list[tid].isActive && !d_particle_list[tid].isRogue) {

      d_particle_list[tid].tau = { 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f };
      d_particle_list[tid].fluxDiv = { 0.0f, 0.0f, 0.0f };

      solve(d_particle_list, tid, 1.0f, 0.0000001f, 10.0f);
      advect(d_particle_list, tid, 1.0f);
    }
  }
  return;
}

__global__ void testCUDA_advection(int length, particle_SOA d_particle_list)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int tid = index; tid < length; tid += stride) {
    // PGD[tid].interp[tid].interpValues(TGD, p[tid].pos, p[tid].tau,p[tid].flux_div, p[tid].nuT, p[tid].CoEps);
    if (d_particle_list.state[tid] == ACTIVE) {

      d_particle_list.tau[tid] = { 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f };
      d_particle_list.flux_div[tid] = { 0.0f, 0.0f, 0.0f };


      solve(d_particle_list, tid, 1.0f, 0.0000001f, 10.0f);
      advect(d_particle_list, tid, 1.0f);
    }
  }
  return;
}

void print_particle(const particle_AOS &p)
{
  std::cout << "--------------------------------------" << std::endl;
  std::cout << "Particle test print:" << std::endl;
  std::cout << " state   : " << p.isActive << ", " << p.isRogue << std::endl;
  std::cout << " position: " << p.pos._1 << ", " << p.pos._2 << ", " << p.pos._3 << std::endl;
  std::cout << " velocity: " << p.velMean._1 << ", " << p.velMean._2 << ", " << p.velMean._3 << std::endl;
  std::cout << " fluct   : " << p.velFluct._1 << ", " << p.velFluct._2 << ", " << p.velFluct._3 << std::endl;
  std::cout << "--------------------------------------" << std::endl;
}

void test_gpu_AOS(const int &length)
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

  particle_AOS tmp;

  tmp.isRogue = false;
  tmp.isActive = true;
  tmp.CoEps = 0.1f;
  tmp.pos = { 0.0f, 0.0f, 0.0f };
  tmp.velMean = { 1.0f, 2.0f, -1.0f };
  tmp.velFluct = { 0.0f, 0.0f, 0.0f };
  tmp.tau = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
  tmp.fluxDiv = { 0.0f, 0.0f, 0.0f };

  std::vector<particle_AOS> particle_list;
  particle_list.resize(length, tmp);

  if (errorCheck == cudaSuccess) {
    // temp
    print_particle(particle_list[0]);

    particle_AOS *d_particle_list;
    cudaMalloc((void **)&d_particle_list, length * sizeof(particle_AOS));


    auto gpuStartTime = std::chrono::high_resolution_clock::now();

    // copy to the device
    cudaMemcpy(d_particle_list, particle_list.data(), length * sizeof(particle_AOS), cudaMemcpyHostToDevice);

    // call kernel
    auto kernelStartTime = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < 1E4; ++k) {
      testCUDA_advection<<<numberOfBlocks, numberOfThreadsPerBlock>>>(length, d_particle_list);
      cudaDeviceSynchronize();
    }
    auto kernelEndTime = std::chrono::high_resolution_clock::now();

    // cudamemcpy back to host
    cudaMemcpy(particle_list.data(), d_particle_list, length * sizeof(particle_AOS), cudaMemcpyDeviceToHost);

    auto gpuEndTime = std::chrono::high_resolution_clock::now();

    // cudafree
    cudaFree(d_particle_list);

    std::chrono::duration<double> kernelElapsed = kernelEndTime - kernelStartTime;
    std::cout << "kernel  elapsed time: " << kernelElapsed.count() << " s\n";
    std::chrono::duration<double> gpuElapsed = gpuEndTime - gpuStartTime;
    std::cout << "GPU  elapsed time: " << gpuElapsed.count() << " s\n";

    print_particle(particle_list[0]);

  } else {
    printf("CUDA ERROR!\n");
  }
}

void test_gpu_SOA(const int &length)
{
  int gpuID = 0;
  cudaError_t errorCheck = cudaGetDevice(&gpuID);

  int blockCount = 1;
  cudaDeviceGetAttribute(&blockCount, cudaDevAttrMultiProcessorCount, gpuID);
  // std::cout << blockCount << std::endl;

  int threadsPerBlock = 128;
  cudaDeviceGetAttribute(&threadsPerBlock, cudaDevAttrMaxThreadsPerBlock, gpuID);
  // std::cout << threadsPerBlock << std::endl;

  int blockSize = threadsPerBlock;
  dim3 numberOfThreadsPerBlock(blockSize, 1, 1);
  dim3 numberOfBlocks(ceil(length / (float)(blockSize)), 1, 1);

  particle_AOS tmp;

  tmp.isRogue = false;
  tmp.isActive = true;
  tmp.CoEps = 0.1f;
  tmp.pos = { 0.0f, 0.0f, 0.0f };
  tmp.velMean = { 1.0f, 2.0f, -1.0f };
  tmp.velFluct = { 0.0f, 0.0f, 0.0f };
  tmp.tau = { 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f };
  tmp.fluxDiv = { 0.0f, 0.0f, 0.0f };


  std::vector<particle_AOS> particle_list;
  particle_list.resize(length, tmp);

  /*std::vector<bool> isRogue(particle_list.size());
  for (size_t k = 0; k < particle_list.size(); ++k) {
    isRogue[k] = particle_list[k].isRogue;
  }
  std::vector<bool> isActive(particle_list.size());
  for (size_t k = 0; k < particle_list.size(); ++k) {
    isActive[k] = particle_list[k].isActive;
  }*/
  std::vector<int> particle_state(particle_list.size());
  for (size_t k = 0; k < particle_list.size(); ++k) {
    if (particle_list[k].isActive)
      particle_state[k] = ACTIVE;
    else if (particle_list[k].isActive && !particle_list[k].isRogue)
      particle_state[k] = INACTIVE;
    else
      particle_state[k] = ROGUE;
  }

  std::vector<float> CoEps(particle_list.size());
  for (size_t k = 0; k < particle_list.size(); ++k) {
    CoEps[k] = particle_list[k].CoEps;
  }

  std::vector<vec3> pos(particle_list.size());
  for (size_t k = 0; k < particle_list.size(); ++k) {
    pos[k] = particle_list[k].pos;
  }
  std::vector<vec3> velMean(particle_list.size());
  for (size_t k = 0; k < particle_list.size(); ++k) {
    velMean[k] = particle_list[k].velMean;
  }
  std::vector<vec3> velFluct(particle_list.size());
  for (size_t k = 0; k < particle_list.size(); ++k) {
    velFluct[k] = particle_list[k].velFluct;
  }
  std::vector<mat3sym> tau(particle_list.size());
  for (size_t k = 0; k < particle_list.size(); ++k) {
    tau[k] = particle_list[k].tau;
  }

  if (errorCheck == cudaSuccess) {
    // temp
    print_particle(particle_list[0]);

    particle_SOA d_particle_list;

    // cudaMalloc((void **)&d_particle_list.isRogue, length * sizeof(bool));
    // cudaMalloc((void **)&d_particle_list.isActive, length * sizeof(bool));

    cudaMalloc((void **)&d_particle_list.state, length * sizeof(int));

    cudaMalloc((void **)&d_particle_list.pos, length * sizeof(vec3));
    cudaMalloc((void **)&d_particle_list.velMean, length * sizeof(vec3));

    cudaMalloc((void **)&d_particle_list.velFluct, length * sizeof(vec3));
    cudaMalloc((void **)&d_particle_list.velFluct_old, length * sizeof(vec3));
    cudaMalloc((void **)&d_particle_list.delta_velFluct, length * sizeof(vec3));

    cudaMalloc((void **)&d_particle_list.CoEps, length * sizeof(float));
    cudaMalloc((void **)&d_particle_list.tau, length * sizeof(mat3sym));
    cudaMalloc((void **)&d_particle_list.tau_old, length * sizeof(mat3sym));

    cudaMalloc((void **)&d_particle_list.flux_div, length * sizeof(vec3));

    auto gpuStartTime = std::chrono::high_resolution_clock::now();

    // copy to the device
    // cudaMemcpy(d_particle_list.isRogue, isRogue.data(), length * sizeof(bool), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_particle_list.isActive, isActive.data(), length * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particle_list.state, particle_state.data(), length * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_particle_list.CoEps, CoEps.data(), length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particle_list.tau, tau.data(), length * sizeof(mat3sym), cudaMemcpyHostToDevice);

    cudaMemcpy(d_particle_list.pos, pos.data(), length * sizeof(vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particle_list.velMean, velMean.data(), length * sizeof(vec3), cudaMemcpyHostToDevice);


    // call kernel
    auto kernelStartTime = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < 1E4; ++k) {
      testCUDA_advection<<<numberOfBlocks, numberOfThreadsPerBlock>>>(length, d_particle_list);
      cudaDeviceSynchronize();
    }
    auto kernelEndTime = std::chrono::high_resolution_clock::now();

    // cudamemcpy back to host
    // cudaMemcpy(isRogue.data(), d_particle_list.isRogue, length * sizeof(bool), cudaMemcpyDeviceToHost);
    // cudaMemcpy(isActive.data(), &d_particle_list.isActive, length * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(particle_state.data(), d_particle_list.state, length * sizeof(int), cudaMemcpyDeviceToHost);

    // cudaMemcpy(CoEps.data(), d_particle_list.CoEps, length * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(pos.data(), d_particle_list.pos, length * sizeof(vec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(velMean.data(), d_particle_list.velMean, length * sizeof(vec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(velFluct.data(), d_particle_list.velFluct, length * sizeof(vec3), cudaMemcpyDeviceToHost);


    for (size_t k = 0; k < particle_list.size(); ++k) {
      particle_list[k].pos = pos[k];
    }
    for (size_t k = 0; k < particle_list.size(); ++k) {
      particle_list[k].velMean = velMean[k];
    }
    for (size_t k = 0; k < particle_list.size(); ++k) {
      particle_list[k].velFluct = velFluct[k];
    }

    for (size_t k = 0; k < particle_list.size(); ++k) {
      if (particle_state[k] == ACTIVE) {
        particle_list[k].isActive = true;
        particle_list[k].isRogue = false;
      } else if (particle_state[k] == INACTIVE) {
        particle_list[k].isActive = false;
        particle_list[k].isRogue = false;
      } else if (particle_state[k] == ROGUE) {
        particle_list[k].isRogue = true;
        particle_list[k].isActive = false;
      } else {
        // unknown state -> should not happend
      }
    }

    auto gpuEndTime = std::chrono::high_resolution_clock::now();

    // cudafree
    cudaFree(d_particle_list.state);

    cudaFree(d_particle_list.CoEps);

    cudaFree(d_particle_list.pos);
    cudaFree(d_particle_list.velMean);

    cudaFree(d_particle_list.velFluct);
    cudaFree(d_particle_list.velFluct_old);
    cudaFree(d_particle_list.delta_velFluct);

    cudaFree(d_particle_list.tau);
    cudaFree(d_particle_list.tau_old);

    cudaFree(d_particle_list.flux_div);

    std::chrono::duration<double> kernelElapsed = kernelEndTime - kernelStartTime;
    std::cout << "kernel  elapsed time: " << kernelElapsed.count() << " s\n";
    std::chrono::duration<double> gpuElapsed = gpuEndTime - gpuStartTime;
    std::cout << "GPU  elapsed time: " << gpuElapsed.count() << " s\n";

    print_particle(particle_list[0]);

  } else {
    printf("CUDA ERROR!\n");
  }
}
