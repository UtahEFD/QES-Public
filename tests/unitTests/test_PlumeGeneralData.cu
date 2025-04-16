/****************************************************************************
 * Copyright (c) 2025 University of Utah
 * Copyright (c) 2025 University of Minnesota Duluth
 *
 * Copyright (c) 2025 Behnam Bozorgmehr
 * Copyright (c) 2025 Jeremy A. Gibbs
 * Copyright (c) 2025 Fabien Margairaz
 * Copyright (c) 2025 Eric R. Pardyjak
 * Copyright (c) 2025 Zachary Patterson
 * Copyright (c) 2025 Rob Stoll
 * Copyright (c) 2025 Lucas Ulmer
 * Copyright (c) 2025 Pete Willemsen
 *
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/**
 * @file vectorMath.cu
 * @brief :document this:
 */

#include "test_PlumeGeneralData.h"
#include "AdvectParticle.cu"

typedef struct
{
  bool isRogue;
  bool isActive;
  vec3 pos;

  vec3 uMean;
  vec3 uFluct;
  vec3 uFluct_old;
  vec3 uFluct_delta;

  mat3sym tau;
  mat3sym tau_old;
  mat3sym tau_ddt;

  vec3 fluxDiv;

} part;


__global__ void testCUDA_matmult(int length, mat3 *d_A, vec3 *d_b, vec3 *d_x)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int it = index; it < length; it += stride) {
    bool tt = invert3(d_A[it]);
    matmult(d_A[it], d_b[it], d_x[it]);
  }
  return;
}

__global__ void testCUDA_invar(int length, mat3sym *d_tau, vec3 *d_invar)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int it = index; it < length; it += stride) {
    makeRealizable(10e-4, d_tau[it]);
    calcInvariants(d_tau[it], d_invar[it]);
  }
  return;
}

__global__ void testCUDA_advection(int length, vec3 *d_pPos)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int it = index; it < length; it += stride) {
    advectParticle(1.0, d_pPos[it]);
  }
  return;
}

void test_PlumeGeneralData::testGPU_struct(int length)
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

  mat3 tmp = { 1, 2, 3, 2, 1, 2, 3, 2, 1 };
  std::vector<mat3> A;
  A.resize(length, tmp);

  std::vector<vec3> b;
  b.resize(length, { 1.0, 1.0, 1.0 });

  std::vector<vec3> x;
  x.resize(length, { 0.0, 0.0, 0.0 });

  std::vector<mat3sym> tau;
  // tau.resize(length, { 1, 2, 3, 1, 2, 1 });
  tau.resize(length, { 1, 0, 3, 0, 0, 1 });
  std::vector<vec3> invar;
  invar.resize(length, { 0.0, 0.0, 0.0 });

  if (errorCheck == cudaSuccess) {
    // temp

    mat3 *d_A;
    cudaMalloc((void **)&d_A, 9 * length * sizeof(float));
    vec3 *d_b;
    cudaMalloc((void **)&d_b, 3 * length * sizeof(float));
    vec3 *d_x;
    cudaMalloc((void **)&d_x, length * sizeof(vec3));

    mat3sym *d_tau;
    cudaMalloc((void **)&d_tau, length * sizeof(mat3sym));
    vec3 *d_invar;
    cudaMalloc((void **)&d_invar, length * sizeof(vec3));


    auto gpuStartTime = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d_A, A.data(), 9 * length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), 3 * length * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_tau, tau.data(), length * sizeof(mat3sym), cudaMemcpyHostToDevice);

    // call kernel
    auto kernelStartTime = std::chrono::high_resolution_clock::now();
    testCUDA_matmult<<<numberOfBlocks, numberOfThreadsPerBlock>>>(length, d_A, d_b, d_x);
    testCUDA_invar<<<numberOfBlocks, numberOfThreadsPerBlock>>>(length, d_tau, d_invar);
    // testCUDA_advection<<<numberOfBlocks, numberOfThreadsPerBlock>>>(length, d_x);
    cudaDeviceSynchronize();
    auto kernelEndTime = std::chrono::high_resolution_clock::now();


    // cudamemcpy back to host
    cudaMemcpy(A.data(), d_A, 9 * length * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(x.data(), d_x, 3 * length * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(tau.data(), d_tau, length * sizeof(mat3sym), cudaMemcpyDeviceToHost);
    cudaMemcpy(invar.data(), d_invar, length * sizeof(vec3), cudaMemcpyDeviceToHost);

    auto gpuEndTime = std::chrono::high_resolution_clock::now();

    // cudafree
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);

    cudaFree(d_tau);
    cudaFree(d_invar);

    std::chrono::duration<double> kernelElapsed = kernelEndTime - kernelStartTime;
    std::cout << "kernel  elapsed time: " << kernelElapsed.count() << " s\n";
    std::chrono::duration<double> gpuElapsed = gpuEndTime - gpuStartTime;
    std::cout << "GPU  elapsed time: " << gpuElapsed.count() << " s\n";

    std::cout << A[0]._11 << " " << A[0]._12 << " " << A[0]._13 << std::endl;
    std::cout << A[0]._21 << " " << A[0]._22 << " " << A[0]._23 << std::endl;
    std::cout << A[0]._31 << " " << A[0]._32 << " " << A[0]._33 << std::endl;
    std::cout << x[0]._1 << " " << x[0]._2 << " " << x[0]._3 << std::endl;

    std::cout << std::endl;
    std::cout << tau[0]._11 << " " << tau[0]._12 << " " << tau[0]._13 << std::endl;
    std::cout << tau[0]._12 << " " << tau[0]._22 << " " << tau[0]._23 << std::endl;
    std::cout << tau[0]._13 << " " << tau[0]._23 << " " << tau[0]._33 << std::endl;
    std::cout << invar[0]._1 << " " << invar[0]._2 << " " << invar[0]._3 << std::endl;


  } else {
    printf("CUDA ERROR!\n");
  }
}
