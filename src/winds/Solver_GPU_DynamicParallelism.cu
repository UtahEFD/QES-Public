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

/** @file Solver_GPU_DynamicParallelism.cu */

#include "Solver_GPU_DynamicParallelism.h"

using namespace std::chrono;
using namespace std;
using std::ofstream;
using std::ifstream;
using std::istringstream;
using std::string;
using std::cerr;
using std::endl;
using std::vector;
using std::cout;
using std::to_string;

#define BLOCKSIZE 1024
#define cudaCheck(x) _cudaCheck(x, #x, __FILE__, __LINE__)

__device__ float error;


template<typename T>
void Solver_GPU_DynamicParallelism::_cudaCheck(T e, const char *func, const char *call, const int line)
{
  if (e != cudaSuccess) {
    printf("\"%s\" at %d in %s\n\treturned %d\n-> %s\n", func, line, call, (int)e, cudaGetErrorString(e));
    exit(EXIT_FAILURE);
  }
}

/// Divergence CUDA Kernel.
/// The divergence kernel ...
///
__global__ void divergence(float *d_u0, float *d_v0, float *d_w0, float *d_R, float *d_e, float *d_f, float *d_g, float *d_h, float *d_m, float *d_n, int alpha1, int nx, int ny, int nz, float dx, float dy, float *d_dz_array)
{

  int icell_cent = blockDim.x * blockIdx.x + threadIdx.x;
  int k = icell_cent / ((nx - 1) * (ny - 1));
  int j = (icell_cent - k * (nx - 1) * (ny - 1)) / (nx - 1);
  int i = icell_cent - k * (nx - 1) * (ny - 1) - j * (nx - 1);
  int icell_face = i + j * nx + k * nx * ny;

  // Would be nice to figure out how to not have this branch check...
  if ((i < nx - 1) && (j < ny - 1) && (k < nz - 1) && (i >= 0) && (j >= 0) && (k > 0)) {

    // Divergence equation
    d_R[icell_cent] = (-2 * pow(alpha1, 2.0)) * (((d_e[icell_cent] * d_u0[icell_face + 1] - d_f[icell_cent] * d_u0[icell_face]) * dx) + ((d_g[icell_cent] * d_v0[icell_face + nx] - d_h[icell_cent] * d_v0[icell_face]) * dy) + (d_m[icell_cent] * d_dz_array[k] * 0.5 * (d_dz_array[k] + d_dz_array[k + 1]) * d_w0[icell_face + nx * ny] - d_n[icell_cent] * d_w0[icell_face] * d_dz_array[k] * 0.5 * (d_dz_array[k] + d_dz_array[k - 1])));
  }
}


/// SOR Red-Black Kernel.
///
///
__global__ void SOR_RB(float *d_lambda, float *d_lambda_old, int nx, int ny, int nz, float omega, float A, float B, float dx, float *d_e, float *d_f, float *d_g, float *d_h, float *d_m, float *d_n, float *d_R, int offset)
{
  int icell_cent = blockDim.x * blockIdx.x + threadIdx.x;
  int k = icell_cent / ((nx - 1) * (ny - 1));
  int j = (icell_cent - k * (nx - 1) * (ny - 1)) / (nx - 1);
  int i = icell_cent - k * (nx - 1) * (ny - 1) - j * (nx - 1);

  if ((i > 0) && (i < nx - 2) && (j > 0) && (j < ny - 2) && (k < nz - 2) && (k > 0) && ((i + j + k) % 2) == offset) {

    d_lambda[icell_cent] = (omega / (d_e[icell_cent] + d_f[icell_cent] + d_g[icell_cent] + d_h[icell_cent] + d_m[icell_cent] + d_n[icell_cent])) * (d_e[icell_cent] * d_lambda[icell_cent + 1] + d_f[icell_cent] * d_lambda[icell_cent - 1] + d_g[icell_cent] * d_lambda[icell_cent + (nx - 1)] + d_h[icell_cent] * d_lambda[icell_cent - (nx - 1)] + d_m[icell_cent] * d_lambda[icell_cent + (nx - 1) * (ny - 1)] + d_n[icell_cent] * d_lambda[icell_cent - (nx - 1) * (ny - 1)] - d_R[icell_cent]) + (1.0 - omega) * d_lambda[icell_cent];/// SOR formulation
  }
}

__global__ void saveLambda(float *d_lambda, float *d_lambda_old, int nx, int ny, int nz)
{
  int ii = blockDim.x * blockIdx.x + threadIdx.x;

  if (ii < (nz - 1) * (ny - 1) * (nx - 1)) {
    d_lambda_old[ii] = d_lambda[ii];
  }
}

__global__ void applyNeumannBC(float *d_lambda, int nx, int ny)
{
  // Neumann boundary condition (lambda (@k=0) = lambda (@k=1))
  int ii = blockDim.x * blockIdx.x + threadIdx.x;

  if (ii < nx * ny) {
    d_lambda[ii] = d_lambda[ii + 1 * (nx - 1) * (ny - 1)];
  }
}

__global__ void calculateError(float *d_lambda, float *d_lambda_old, int nx, int ny, int nz, float *d_value, float *d_bvalue)
{
  int d_size = (nx - 1) * (ny - 1) * (nz - 1);
  int ii = blockDim.x * blockIdx.x + threadIdx.x;
  int numblocks = (d_size / BLOCKSIZE) + 1;

  if (ii < d_size) {
    d_value[ii] = fabs(d_lambda[ii] - d_lambda_old[ii]);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    d_bvalue[blockIdx.x] = 0.0;
    for (int j = 0; j < BLOCKSIZE; j++) {
      int index = blockIdx.x * blockDim.x + j;
      if (index < d_size) {
        if (d_value[index] > d_bvalue[blockIdx.x]) {
          d_bvalue[blockIdx.x] = d_value[index];
        }
      }
    }
  }

  // final step of the reduction is made in the mother kernel
  return;
}

// Euler Final Velocity kernel
__global__ void finalVelocity(float *d_lambda, float *d_u, float *d_v, float *d_w, int *d_icellflag, float *d_f, float *d_h, float *d_n, int alpha1, int alpha2, float dx, float dy, float dz, float *d_dz_array, int nx, int ny, int nz)
{

  int icell_face = blockDim.x * blockIdx.x + threadIdx.x;
  int k = icell_face / (nx * ny);
  int j = (icell_face - k * nx * ny) / nx;
  int i = icell_face - k * nx * ny - j * nx;
  int icell_cent = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);/// Lineralized index for cell centered values


  if ((i > 0) && (i < nx - 1) && (j > 0) && (j < ny - 1) && (k < nz - 2) && (k > 0)) {

    d_u[icell_face] = d_u[icell_face] + (1 / (2 * pow(alpha1, 2.0))) * d_f[icell_cent] * dx * (d_lambda[icell_cent] - d_lambda[icell_cent - 1]);
    d_v[icell_face] = d_v[icell_face] + (1 / (2 * pow(alpha1, 2.0))) * d_h[icell_cent] * dy * (d_lambda[icell_cent] - d_lambda[icell_cent - (nx - 1)]);
    d_w[icell_face] = d_w[icell_face] + (1 / (2 * pow(alpha2, 2.0))) * d_n[icell_cent] * d_dz_array[k] * (d_lambda[icell_cent] - d_lambda[icell_cent - (nx - 1) * (ny - 1)]);
  }


  if ((i >= 0) && (i < nx - 1) && (j >= 0) && (j < ny - 1) && (k < nz - 1) && (k >= 1) && ((d_icellflag[icell_cent] == 0) || (d_icellflag[icell_cent] == 2))) {
    d_u[icell_face] = 0;
    d_u[icell_face + 1] = 0;
    d_v[icell_face] = 0;
    d_v[icell_face + nx] = 0;
    d_w[icell_face] = 0;
    d_w[icell_face + nx * ny] = 0;
  }
}


/// SOR iteration kernel
///
__global__ void SOR_iteration(float *d_lambda, float *d_lambda_old, int nx, int ny, int nz, float omega, float A, float B, float dx, float dy, float dz, float *d_dz_array, float *d_e, float *d_f, float *d_g, float *d_h, float *d_m, float *d_n, float *d_R, int itermax, float tol, float *d_value, float *d_bvalue, int alpha1, int alpha2, float *d_u, float *d_v, float *d_w, int *d_icellflag)
{
  int iter = 0;
  error = 1.0;

  // Calculate divergence of initial velocity field
  dim3 numberOfThreadsPerBlock(BLOCKSIZE, 1, 1);
  dim3 numberOfBlocks(ceil(((nx - 1) * (ny - 1) * (nz - 1)) / (float)(BLOCKSIZE)), 1, 1);

  // Invoke divergence kernel
  if (itermax > 0) {
    divergence<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_u, d_v, d_w, d_R, d_e, d_f, d_g, d_h, d_m, d_n, alpha1, nx, ny, nz, dx, dy, d_dz_array);
  }

  // Iterate untill convergence is reached
  while ((iter < itermax) && (error > tol) && (itermax > 0)) {

    // Save previous iteration values for error calculation
    saveLambda<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_lambda, d_lambda_old, nx, ny, nz);
    // cudaDeviceSynchronize();
    __syncthreads();
    // SOR part
    int offset = 0;// red nodes
    //  Invoke red-black SOR kernel for red nodes
    SOR_RB<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_lambda, d_lambda_old, nx, ny, nz, omega, A, B, dx, d_e, d_f, d_g, d_h, d_m, d_n, d_R, offset);
    // cudaDeviceSynchronize();
    __syncthreads();
    offset = 1;// black nodes
    //  Invoke red-black SOR kernel for black nodes
    SOR_RB<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_lambda, d_lambda_old, nx, ny, nz, omega, A, B, dx, d_e, d_f, d_g, d_h, d_m, d_n, d_R, offset);
    // cudaDeviceSynchronize();
    __syncthreads();

    dim3 numberOfBlocks2(ceil(((nx - 1) * (ny - 1)) / (float)(BLOCKSIZE)), 1, 1);
    // Invoke kernel to apply Neumann boundary condition (lambda (@k=0) = lambda (@k=1))
    applyNeumannBC<<<numberOfBlocks2, numberOfThreadsPerBlock>>>(d_lambda, nx, ny);
    // cudaDeviceSynchronize();
    __syncthreads();
    // Error calculation
    calculateError<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_lambda, d_lambda_old, nx, ny, nz, d_value, d_bvalue);
    // cudaDeviceSynchronize();
    __syncthreads();
    // Final step of the reduction to calculate error.
    error = 0.0;
    for (int k = 0; k < ((nx - 1) * (ny - 1) * (nz - 1) / BLOCKSIZE) + 1; k++) {
      if (d_bvalue[k] > error) {
        error = d_bvalue[k];
      }
    }

    iter += 1;
  }

  dim3 numberOfBlocks3(ceil((nx * ny * nz) / (float)(BLOCKSIZE)), 1, 1);
  // Invoke final velocity (Euler) kernel
  if (itermax > 0) {
    finalVelocity<<<numberOfBlocks3, numberOfThreadsPerBlock>>>(d_lambda, d_u, d_v, d_w, d_icellflag, d_f, d_h, d_n, alpha1, alpha2, dx, dy, dz, d_dz_array, nx, ny, nz);
  }

  printf("[Solver] Residual after %d itertations: %2.9f\n", iter, error);
  // printf("Error = %2.9f\n", error);
  // printf("Number of iteration = %d\n", iter);
}


Solver_GPU_DynamicParallelism::Solver_GPU_DynamicParallelism(qes::Domain domain_in, const float &tolerance)
  : Solver(std::move(domain_in), tolerance)
{
  std::cout << "-------------------------------------------------------------------" << std::endl;
  std::cout << "[Solver]\t Initialoization of Dynamic Parallelism Solver" << std::endl;
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    std::cerr << "ERROR!   cudaGetDeviceCount returned "
              << static_cast<int>(error_id) << "\n\t-> "
              << cudaGetErrorString(error_id) << std::endl;
    exit(EXIT_FAILURE);
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    std::cerr << "There are no available device(s) that support CUDA\n";
    exit(EXIT_FAILURE);
  } else {
    std::cout << "\tDetected " << deviceCount << " CUDA Capable device(s)" << std::endl;
  }

  int dev, driverVersion = 0, runtimeVersion = 0;

  for (dev = 0; dev < deviceCount; ++dev) {

    cudaSetDevice(dev);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    std::cout << "\tDevice " << dev << ": " << deviceProp.name << std::endl;

    // Console log
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    std::cout << "\t\tCUDA Driver Version / Runtime Version: "
              << driverVersion / 1000 << "." << (driverVersion % 100) / 10 << " / "
              << runtimeVersion / 1000 << "." << (runtimeVersion % 100) / 10 << std::endl;

    std::cout << "\t\tCUDA Capability Major/Minor version number: "
              << deviceProp.major << "." << deviceProp.minor << std::endl;

    char msg[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(msg, sizeof(msg),
              "\t\tTotal amount of global memory: %.0f MBytes "
              "(%llu bytes)\n",
              static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
              (unsigned long long)deviceProp.totalGlobalMem);
#else
    snprintf(msg, sizeof(msg),
             "\t\tTotal amount of global memory: %.0f MBytes "
             "(%llu bytes)\n",
             static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
             (unsigned long long)deviceProp.totalGlobalMem);
#endif
    std::cout << msg;

    //    printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
    //           deviceProp.multiProcessorCount,
    //           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
    //           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
    //           deviceProp.multiProcessorCount);

    std::cout << "\t\tGPU Max Clock rate:  "
              << deviceProp.clockRate * 1e-3f << " MHz ("
              << deviceProp.clockRate * 1e-6f << " GHz)" << std::endl;

    std::cout << "\t\tPCI: BusID=" << deviceProp.pciBusID << ", "
              << "DeviceID=" << deviceProp.pciDeviceID << ", "
              << "DomainID=" << deviceProp.pciDomainID << std::endl;
  }
  cudaSetDevice(0);

#if 0
  char msg[256];
  int numblocks = (WGD->numcell_cent / BLOCKSIZE) + 1;
  long long memory_req = (10 * numcell_cent + 6 * WGD->numcell_face + numblocks + (WGD->nz - 1)) * sizeof(float)
                         + (WGD->numcell_cent) * sizeof(int) + 2308964352;

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  sprintf_s(msg, sizeof(msg),
            "Total global memory required for running this case: %.0f MBytes "
            "(%llu bytes)",
            static_cast<float>(memory_req / 1048576.0f),
            (unsigned long long)memory_req);
#else
  snprintf(msg, sizeof(msg),
           "Total global memory required for running this case: %.0f MBytes "
           "(%llu bytes)",
           static_cast<float>(memory_req / 1048576.0f),
           (unsigned long long)memory_req);
#endif
  std::cout << "[Solver] " << msg << std::endl;
#endif
}


void Solver_GPU_DynamicParallelism::solve(WINDSGeneralData *WGD, const int &itermax)
{
  auto [nx, ny, nz] = domain.getDomainCellNum();
  auto [dx, dy, dz] = domain.getDomainSize();
  long numcell_face = domain.numFaceCentered();
  long numcell_cent = domain.numCellCentered();

  int numblocks = (numcell_cent / BLOCKSIZE) + 1;

  std::vector<float> value(numcell_cent, 0.0);
  std::vector<float> bvalue(numblocks, 0.0);
  // float *d_u0, *d_v0, *d_w0;
  float *d_value, *d_bvalue;
  float *d_u, *d_v, *d_w;
  int *d_icellflag;
  float *d_dz_array;

  std::cout << "[Solver]\t Running Dynamic Parallelim Solver (GPU) ..." << std::endl;

  auto start = std::chrono::high_resolution_clock::now();// Start recording execution time

  cudaMalloc((void **)&d_e, numcell_cent * sizeof(float));
  cudaMalloc((void **)&d_f, numcell_cent * sizeof(float));
  cudaMalloc((void **)&d_g, numcell_cent * sizeof(float));
  cudaMalloc((void **)&d_h, numcell_cent * sizeof(float));
  cudaMalloc((void **)&d_m, numcell_cent * sizeof(float));
  cudaMalloc((void **)&d_n, numcell_cent * sizeof(float));
  cudaMalloc((void **)&d_R, numcell_cent * sizeof(float));
  cudaMalloc((void **)&d_lambda, numcell_cent * sizeof(float));
  cudaMalloc((void **)&d_lambda_old, numcell_cent * sizeof(float));
  cudaMalloc((void **)&d_icellflag, numcell_cent * sizeof(int));
  cudaMalloc((void **)&d_value, numcell_cent * sizeof(float));
  cudaMalloc((void **)&d_bvalue, numblocks * sizeof(float));
  cudaMalloc((void **)&d_dz_array, (nz - 1) * sizeof(float));
  cudaMalloc((void **)&d_u, numcell_face * sizeof(float));
  cudaMalloc((void **)&d_v, numcell_face * sizeof(float));
  cudaMalloc((void **)&d_w, numcell_face * sizeof(float));

#if 0
  long long memory_req = (10 * numcell_cent + 6 * numcell_face + numblocks + (nz - 1)) * sizeof(float) + (numcell_cent) * sizeof(int) + 2308964352;
  char msg[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  sprintf_s(msg, sizeof(msg),
            "  Total global memory required for running this case:                 %.0f MBytes "
            "(%llu bytes)\n",
            static_cast<float>(memory_req / 1048576.0f),
            (unsigned long long)memory_req);
#else
  snprintf(msg, sizeof(msg),
           "  Total global memory required for running this case:                 %.0f MBytes "
           "(%llu bytes)\n",
           static_cast<float>(memory_req / 1048576.0f),
           (unsigned long long)memory_req);
#endif
  std::cout << msg;
#endif

  cudaMemcpy(d_icellflag, WGD->icellflag.data(), numcell_cent * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_u, WGD->u0.data(), numcell_face * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, WGD->v0.data(), numcell_face * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, WGD->w0.data(), numcell_face * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_R, R.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_value, value.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bvalue, bvalue.data(), numblocks * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_e, WGD->e.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_f, WGD->f.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_g, WGD->g.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_h, WGD->h.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_m, WGD->m.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_n, WGD->n.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dz_array, WGD->domain.dz_array.data(), (nz - 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lambda, lambda.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lambda_old, lambda_old.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);

  /////////////////////////////////////////////////
  //                 SOR solver              //////
  /////////////////////////////////////////////////

  // Invoke the main (mother) kernel
<<<<<<< HEAD:src/winds/DynamicParallelism.cu
  SOR_iteration<<<1, 1>>>(d_lambda, d_lambda_old, WGD->nx, WGD->ny, WGD->nz, omega, A, B, WGD->dx, WGD->dy, WGD->dz, d_dz_array, d_e, d_f, d_g, d_h, d_m, d_n, d_R, itermax, tol, d_value, d_bvalue, alpha1, alpha2, d_u, d_v, d_w, d_icellflag);
  // cudaCheck(cudaGetLastError());
=======
  SOR_iteration<<<1, 1>>>(d_lambda, d_lambda_old, nx, ny, nz, omega, A, B, dx, dy, dz, d_dz_array, d_e, d_f, d_g, d_h, d_m, d_n, d_R, itermax, tol, d_value, d_bvalue, alpha1, alpha2, d_u, d_v, d_w, d_icellflag);
  cudaCheck(cudaGetLastError());
>>>>>>> 6aee4dc53fb682adbc547a865dc1f74ace4421cc:src/winds/Solver_GPU_DynamicParallelism.cu

  cudaMemcpy(WGD->u.data(), d_u, numcell_face * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(WGD->v.data(), d_v, numcell_face * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(WGD->w.data(), d_w, numcell_face * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(lambda.data(), d_lambda, numcell_cent * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(lambda_old.data(), d_lambda_old, numcell_cent * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(R.data(), d_R, numcell_cent * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_lambda);
  cudaFree(d_lambda_old);
  cudaFree(d_e);
  cudaFree(d_f);
  cudaFree(d_g);
  cudaFree(d_h);
  cudaFree(d_m);
  cudaFree(d_n);
  cudaFree(d_R);
  cudaFree(d_value);
  cudaFree(d_bvalue);
  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_w);
  cudaFree(d_dz_array);
  cudaFree(d_icellflag);

  auto finish = std::chrono::high_resolution_clock::now();// Finish recording execution time

  std::chrono::duration<float> elapsed = finish - start;
  std::cout << "\t\t Elapsed time: " << elapsed.count() << " s\n";// Print out elapsed execution time
}
