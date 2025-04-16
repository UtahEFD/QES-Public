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
 ***************************************************************************/

/**
 * @file Solver_GPU_GlobalMemory.cu
 * @brief Child class of the Solver that runs the convergence
 * algorithm using Global Memory on a single GPU.
 *
 * @sa Solver
 * @sa Solver_GPU_GlobalMemory
 */

#include "Solver_GPU_GlobalMemory.h"

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


template<typename T>
void Solver_GPU_GlobalMemory::_cudaCheck(T e, const char *func, const char *call, const int line)
{
  if (e != cudaSuccess) {
    printf("\"%s\" at %d in %s\n\treturned %d\n-> %s\n", func, line, call, (int)e, cudaGetErrorString(e));
    exit(EXIT_FAILURE);
  }
}

// Divergence CUDA Kernel.
// The divergence kernel ...
//
__global__ void divergenceGlobal(float *d_u0, float *d_v0, float *d_w0, float *d_R, float *d_e, float *d_f, float *d_g, float *d_h, float *d_m, float *d_n, int alpha1, int nx, int ny, int nz, float dx, float dy, float *d_dz_array)
{

  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int k = id / ((nx - 1) * (ny - 1));
  int j = (id - k * (nx - 1) * (ny - 1)) / (nx - 1);
  int i = id - k * (nx - 1) * (ny - 1) - j * (nx - 1);
  int icell_face = i + j * nx + k * nx * ny;

  // Would be nice to figure out how to not have this branch check...
  if ((i < nx - 1) && (j < ny - 1) && (k < nz - 1) && (i >= 0) && (j >= 0) && (k > 0)) {

    // Divergence equation
    d_R[id] = (-2 * pow(alpha1, 2.0))
              * (((d_e[id] * d_u0[icell_face + 1] - d_f[id] * d_u0[icell_face]) * dx)
                 + ((d_g[id] * d_v0[icell_face + nx] - d_h[id] * d_v0[icell_face]) * dy)
                 + (d_m[id] * d_dz_array[k] * 0.5 * (d_dz_array[k] + d_dz_array[k + 1]) * d_w0[icell_face + nx * ny]
                    - d_n[id] * d_w0[icell_face] * d_dz_array[k] * 0.5 * (d_dz_array[k] + d_dz_array[k - 1])));
  }
}


// SOR RedBlack Kernel.
//
//
__global__ void SOR_RB_Global(float *d_lambda, int nx, int ny, int nz, float omega, float A, float B, float *d_e, float *d_f, float *d_g, float *d_h, float *d_m, float *d_n, float *d_R, int offset)
{
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int k = id / ((nx - 1) * (ny - 1));
  int j = (id - k * (nx - 1) * (ny - 1)) / (nx - 1);
  int i = id - k * (nx - 1) * (ny - 1) - j * (nx - 1);

  if ((i > 0) && (i < nx - 2) && (j > 0) && (j < ny - 2) && (k < nz - 2) && (k > 0)
      && ((i + j + k) % 2) == offset) {
    // SOR formulation
    d_lambda[id] = (omega / (d_e[id] + d_f[id] + d_g[id] + d_h[id] + d_m[id] + d_n[id]))
                     * (d_e[id] * d_lambda[id + 1]
                        + d_f[id] * d_lambda[id - 1]
                        + d_g[id] * d_lambda[id + (nx - 1)]
                        + d_h[id] * d_lambda[id - (nx - 1)]
                        + d_m[id] * d_lambda[id + (nx - 1) * (ny - 1)]
                        + d_n[id] * d_lambda[id - (nx - 1) * (ny - 1)]
                        - d_R[id])
                   + (1.0 - omega) * d_lambda[id];
  }
}


__global__ void saveLambdaGlobal(float *d_lambda, float *d_lambda_old, int d_size)
{
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  if (id < d_size) {
    d_lambda_old[id] = d_lambda[id];
  }
}

__global__ void applyNeumannBCGlobal(float *d_lambda, int nx, int ny)
{
  // Neumann boundary condition (lambda (@k=0) = lambda (@k=1))
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  if (id < nx * ny) {
    d_lambda[id] = d_lambda[id + 1 * (nx - 1) * (ny - 1)];
  }
}

__global__ void calculateErrorGlobal2(float *d_lambda, float *d_lambda_old, float *d_value, int d_size)
{
  int ii = blockDim.x * blockIdx.x + threadIdx.x;
  // int numblocks = (d_size / BLOCKSIZE) + 1;

  if (ii < d_size) {
    float error = fabs(d_lambda[ii] - d_lambda_old[ii]);

    // atomicMAX using atomicCAS
    int r = __float_as_int(*d_value);
    while (error > __int_as_float(r)) {
      int o = r;
      if ((r = atomicCAS((int *)d_value, o, __float_as_int(error))) == o)
        break;
    }
    error = __int_as_float(r);
  }
}

// Euler Final Velocity kernel
__global__ void finalVelocityGlobal(float *d_lambda, float *d_u, float *d_v, float *d_w, int *d_icellflag, float *d_f, float *d_h, float *d_n, int alpha1, int alpha2, float dx, float dy, float dz, float *d_dz_array, int nx, int ny, int nz)
{

  int icell_face = blockDim.x * blockIdx.x + threadIdx.x;
  int k = icell_face / (nx * ny);
  int j = (icell_face - k * nx * ny) / nx;
  int i = icell_face - k * nx * ny - j * nx;
  int icell_cent = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);// Lineralized index for cell centered values


  if ((i > 0) && (i < nx - 1) && (j > 0) && (j < ny - 1) && (k < nz - 2) && (k > 0)) {

    d_u[icell_face] = d_u[icell_face]
                      + (1.0 / (2.0 * pow(alpha1, 2.0)))
                          * d_f[icell_cent]
                          * dx
                          * (d_lambda[icell_cent] - d_lambda[icell_cent - 1]);

    d_v[icell_face] = d_v[icell_face]
                      + (1.0 / (2.0 * pow(alpha1, 2.0)))
                          * d_h[icell_cent]
                          * dy
                          * (d_lambda[icell_cent] - d_lambda[icell_cent - (nx - 1)]);

    d_w[icell_face] = d_w[icell_face]
                      + (1.0 / (2.0 * pow(alpha2, 2.0)))
                          * d_n[icell_cent]
                          * d_dz_array[k]
                          * (d_lambda[icell_cent] - d_lambda[icell_cent - (nx - 1) * (ny - 1)]);
  }

  if ((i >= 0) && (i < nx - 1)
      && (j >= 0) && (j < ny - 1)
      && (k < nz - 1) && (k >= 1)
      && ((d_icellflag[icell_cent] == 0) || (d_icellflag[icell_cent] == 2))) {
    d_u[icell_face] = 0;
    d_u[icell_face + 1] = 0;
    d_v[icell_face] = 0;
    d_v[icell_face + nx] = 0;
    d_w[icell_face] = 0;
    d_w[icell_face + nx * ny] = 0;
  }
}

Solver_GPU_GlobalMemory::Solver_GPU_GlobalMemory(qes::Domain domain_in, const float &tolerance)
  : Solver(std::move(domain_in), tolerance)
{
  std::cout << "-------------------------------------------------------------------" << std::endl;
  std::cout << "[Solver]\t Initializing Global Memory Solver (GPU) ..." << std::endl;
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    std::cerr << "\n===================================================================" << std::endl;
    std::cerr << "[ERROR]\t cudaGetDeviceCount returned "
              << static_cast<int>(error_id) << "\n\t-> "
              << cudaGetErrorString(error_id) << std::endl;
    std::cerr << "===================================================================" << std::endl;
    exit(EXIT_FAILURE);
  }

  // This function call returnsq 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    std::cerr << "\n===================================================================" << std::endl;
    std::cerr << "[ERROR]\t There are no available device(s) that support CUDA\n";
    std::cerr << "===================================================================" << std::endl;
    exit(EXIT_FAILURE);
  }

  cudaSetDevice(0);
}

void Solver_GPU_GlobalMemory::solve(WINDSGeneralData *WGD, const int &itermax)
{
  auto [nx, ny, nz] = domain.getDomainCellNum();
  auto [dx, dy, dz] = domain.getDomainSize();
  long numcell_face = domain.numFaceCentered();
  long numcell_cent = domain.numCellCentered();

  // int numblocks = (numcell_cent / BLOCKSIZE) + 1;
  R.resize(numcell_cent, 0.0);

  std::cout << "[Solver] Running Global Memory Solver (GPU) ..." << std::endl;

  // std::vector<float> value(numcell_cent, 0.0);
  // std::vector<float> bvalue(numblocks, 0.0);

  float *d_u, *d_v, *d_w;
  // float *d_value, *d_bvalue;
  int *d_icellflag;
  float *d_dz_array;
  float *d_error;

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

  cudaMalloc((void **)&d_u, numcell_face * sizeof(float));
  cudaMalloc((void **)&d_v, numcell_face * sizeof(float));
  cudaMalloc((void **)&d_w, numcell_face * sizeof(float));

  cudaMalloc((void **)&d_dz_array, (nz - 1) * sizeof(float));
  cudaMalloc((void **)&d_icellflag, numcell_cent * sizeof(int));

  cudaMemcpy(d_u, WGD->u0.data(), numcell_face * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, WGD->v0.data(), numcell_face * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, WGD->w0.data(), numcell_face * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemset(d_R, 0.0, numcell_cent * sizeof(float));
  cudaMemcpy(d_lambda_old, lambda_old.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lambda, lambda.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_e, WGD->e.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_f, WGD->f.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_g, WGD->g.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_h, WGD->h.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_m, WGD->m.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_n, WGD->n.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_dz_array, WGD->domain.dz_array.data(), (nz - 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_icellflag, WGD->icellflag.data(), numcell_cent * sizeof(int), cudaMemcpyHostToDevice);

  dim3 numberOfThreadsPerBlock(BLOCKSIZE, 1, 1);
  dim3 numberOfBlocks(ceil(domain.numCellCentered() / (float)(BLOCKSIZE)), 1, 1);

  // Invoke divergence kernel
  divergenceGlobal<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_u, d_v, d_w, d_R, d_e, d_f, d_g, d_h, d_m, d_n, alpha1, nx, ny, nz, dx, dy, d_dz_array);


  /////////////////////////////////////////////////
  //                 SOR solver              //////
  /////////////////////////////////////////////////

  int iter = 0;
  // float error;
  // std::vector<float> max_error(1, 1.0);
  float max_error = 1.0;

  cudaMalloc((void **)&d_error, 1 * sizeof(float));

  // Main solver loop
  while ((iter < itermax) && (max_error > tol)) {
    // Save previous iteration values for error calculation
    saveLambdaGlobal<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_lambda, d_lambda_old, numcell_cent);
    cudaCheck(cudaGetLastError());

    // Red nodes pass
    SOR_RB_Global<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_lambda, nx, ny, nz, omega, A, B, d_e, d_f, d_g, d_h, d_m, d_n, d_R, 0);
    cudaCheck(cudaGetLastError());

    // Black nodes pass
    SOR_RB_Global<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_lambda, nx, ny, nz, omega, A, B, d_e, d_f, d_g, d_h, d_m, d_n, d_R, 1);
    cudaCheck(cudaGetLastError());

    dim3 numberOfBlocks2(ceil(domain.numHorizontalCellCentered() / (float)(BLOCKSIZE)), 1, 1);
    // Invoke kernel to apply Neumann boundary condition (lambda (@k=0) = lambda (@k=1))
    applyNeumannBCGlobal<<<numberOfBlocks2, numberOfThreadsPerBlock>>>(d_lambda, nx, ny);

    // Calculation of error
    cudaMemset(d_error, 0, sizeof(float));
    calculateErrorGlobal2<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_lambda, d_lambda_old, d_error, numcell_cent);
    cudaMemcpy(&max_error, d_error, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    // incrementation of iteration counter
    iter += 1;
  }

  dim3 numberOfBlocks3(ceil(domain.numFaceCentered() / (float)(BLOCKSIZE)), 1, 1);
  // Invoke final velocity (Euler) kernel
  finalVelocityGlobal<<<numberOfBlocks3, numberOfThreadsPerBlock>>>(d_lambda, d_u, d_v, d_w, d_icellflag, d_f, d_h, d_n, alpha1, alpha2, dx, dy, dz, d_dz_array, nx, ny, nz);
  cudaCheck(cudaGetLastError());

  cudaMemcpy(WGD->u.data(), d_u, numcell_face * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(WGD->v.data(), d_v, numcell_face * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(WGD->w.data(), d_w, numcell_face * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_R);
  cudaFree(d_lambda);
  cudaFree(d_lambda_old);

  cudaFree(d_e);
  cudaFree(d_f);
  cudaFree(d_g);
  cudaFree(d_h);
  cudaFree(d_m);
  cudaFree(d_n);

  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_w);

  cudaFree(d_dz_array);
  cudaFree(d_icellflag);

  auto finish = std::chrono::high_resolution_clock::now();// Finish recording execution time

  std::chrono::duration<float> elapsed = finish - start;
  printf("[Solver]\t Residual after %d itertations: %2.9f\n", iter, max_error);
  // std::cout << "Error:" << max_error[0] << "\n";
  // std::cout << "Number of iterations:" << iter << "\n";// Print the number of iterations
  std::cout << "\t\t elapsed time: " << elapsed.count() << " s\n";// Print out elapsed execution time
}
