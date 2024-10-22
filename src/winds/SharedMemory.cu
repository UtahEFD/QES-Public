/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Behnam Bozorgmehr
 * Copyright (c) 2024 Jeremy A. Gibbs
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Eric R. Pardyjak
 * Copyright (c) 2024 Zachary Patterson
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Lucas Ulmer
 * Copyright (c) 2024 Pete Willemsen
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
 * @file SharedMemory.cu
 * @brief Child class of the Solver that runs the convergence
 * algorithm using DynamicParallelism on a single GPU.
 *
 * @sa Solver
 * @sa DynamicParallelism
 * @sa WINDSInputData
 */

#include "SharedMemory.h"

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
void SharedMemory::_cudaCheck(T e, const char *func, const char *call, const int line)
{
  if (e != cudaSuccess) {
    printf("\"%s\" at %d in %s\n\treturned %d\n-> %s\n", func, line, call, (int)e, cudaGetErrorString(e));
    exit(EXIT_FAILURE);
  }
}

// Divergence CUDA Kernel.
// The divergence kernel ...
//
__global__ void divergenceShared(float *d_u0, float *d_v0, float *d_w0, float *d_R, float *d_e, float *d_f, float *d_g, float *d_h, float *d_m, float *d_n, int alpha1, int nx, int ny, int nz, float dx, float dy, float *d_dz_array)
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


// SOR RedBlack Kernel.
//
//
__global__ void SOR_RB_Shared(float *d_lambda, int nx, int ny, int nz, float omega, float A, float B, float *d_e, float *d_f, float *d_g, float *d_h, float *d_m, float *d_n, float *d_R, int offset)
{
  int icell_cent = blockDim.x * blockIdx.x + threadIdx.x;
  int k = icell_cent / ((nx - 1) * (ny - 1));
  int j = (icell_cent - k * (nx - 1) * (ny - 1)) / (nx - 1);
  int i = icell_cent - k * (nx - 1) * (ny - 1) - j * (nx - 1);

  if ((i > 0) && (i < nx - 2) && (j > 0) && (j < ny - 2) && (k < nz - 2) && (k > 0) && ((i + j + k) % 2) == offset) {

    d_lambda[icell_cent] = (omega / (d_e[icell_cent] + d_f[icell_cent] + d_g[icell_cent] + d_h[icell_cent] + d_m[icell_cent] + d_n[icell_cent])) * (d_e[icell_cent] * d_lambda[icell_cent + 1] + d_f[icell_cent] * d_lambda[icell_cent - 1] + d_g[icell_cent] * d_lambda[icell_cent + (nx - 1)] + d_h[icell_cent] * d_lambda[icell_cent - (nx - 1)] + d_m[icell_cent] * d_lambda[icell_cent + (nx - 1) * (ny - 1)] + d_n[icell_cent] * d_lambda[icell_cent - (nx - 1) * (ny - 1)] - d_R[icell_cent]) + (1.0 - omega) * d_lambda[icell_cent];// SOR formulation
  }
}


__global__ void saveLambdaShared(float *d_lambda, float *d_lambda_old, int nx, int ny, int nz)
{
  int ii = blockDim.x * blockIdx.x + threadIdx.x;

  if (ii < (nz - 1) * (ny - 1) * (nx - 1)) {
    d_lambda_old[ii] = d_lambda[ii];
  }
}

__global__ void applyNeumannBCShared(float *d_lambda, int nx, int ny)
{
  // Neumann boundary condition (lambda (@k=0) = lambda (@k=1))
  int ii = blockDim.x * blockIdx.x + threadIdx.x;

  if (ii < nx * ny) {
    d_lambda[ii] = d_lambda[ii + 1 * (nx - 1) * (ny - 1)];
  }
}

__global__ void calculateErrorShared(float *d_lambda, float *d_lambda_old, int nx, int ny, int nz, float *d_value, float *d_bvalue, float *error)
{

  int d_size = (nx - 1) * (ny - 1) * (nz - 1);
  int ii = blockDim.x * blockIdx.x + threadIdx.x;
  int numblocks = (d_size / BLOCKSIZE) + 1;

  if (ii < d_size) {
    d_value[ii] = fabs(d_lambda[ii] - d_lambda_old[ii]);
  }

  __syncthreads();

  if (threadIdx.x > 0) {
    return;
  }
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


  __syncthreads();


  if (ii > 0) {
    return;
  }

  error[0] = 0.0;

  if (ii == 0) {
    for (int k = 0; k < numblocks; k++) {
      if (d_bvalue[k] > error[0]) {
        error[0] = d_bvalue[k];
      }
    }
  }
}


// Euler Final Velocity kernel
__global__ void finalVelocityShared(float *d_lambda, float *d_u, float *d_v, float *d_w, int *d_icellflag, float *d_f, float *d_h, float *d_n, int alpha1, int alpha2, float dx, float dy, float dz, float *d_dz_array, int nx, int ny, int nz)
{

  int icell_face = blockDim.x * blockIdx.x + threadIdx.x;
  int k = icell_face / (nx * ny);
  int j = (icell_face - k * nx * ny) / nx;
  int i = icell_face - k * nx * ny - j * nx;
  int icell_cent = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);// Lineralized index for cell centered values


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


void SharedMemory::solve(const WINDSInputData *WID, WINDSGeneralData *WGD, bool solveWind)
{
  auto [nx, ny, nz] = domain.getDomainCellNum();
  auto [dx, dy, dz] = domain.getDomainSize();
  long numcell_face = domain.numFaceCentered();
  long numcell_cent = domain.numCellCentered();

  itermax = WID->simParams->maxIterations;
  int numblocks = (numcell_cent / BLOCKSIZE) + 1;
  R.resize(numcell_cent, 0.0);

  std::cout << "[Solver]\t Running Shared Memory Solver (GPU) ..." << std::endl;

  std::vector<float> value(numcell_cent, 0.0);
  std::vector<float> bvalue(numblocks, 0.0);

  float *d_u, *d_v, *d_w;
  float *d_value, *d_bvalue;
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
  cudaMalloc((void **)&d_lambda, numcell_cent * sizeof(float));
  cudaMalloc((void **)&d_lambda_old, numcell_cent * sizeof(float));
  cudaMalloc((void **)&d_dz_array, (nz - 1) * sizeof(float));
  cudaMalloc((void **)&d_R, numcell_cent * sizeof(float));
  cudaMalloc((void **)&d_value, numcell_cent * sizeof(float));
  cudaMalloc((void **)&d_bvalue, numblocks * sizeof(float));
  cudaMalloc((void **)&d_icellflag, numcell_cent * sizeof(int));
  cudaMalloc((void **)&d_u, numcell_face * sizeof(float));
  cudaMalloc((void **)&d_v, numcell_face * sizeof(float));
  cudaMalloc((void **)&d_w, numcell_face * sizeof(float));

  cudaMemcpy(d_u, WGD->u0.data(), numcell_face * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, WGD->v0.data(), numcell_face * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, WGD->w0.data(), numcell_face * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_R, R.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_e, WGD->e.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_f, WGD->f.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_g, WGD->g.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_h, WGD->h.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_m, WGD->m.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_n, WGD->n.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dz_array, domain.dz_array.data(), (nz - 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lambda_old, lambda_old.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_value, value.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bvalue, bvalue.data(), numblocks * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lambda, lambda.data(), numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_icellflag, WGD->icellflag.data(), numcell_cent * sizeof(int), cudaMemcpyHostToDevice);

  dim3 numberOfThreadsPerBlock(BLOCKSIZE, 1, 1);
  dim3 numberOfBlocks(ceil(domain.numCellCentered() / (float)(BLOCKSIZE)), 1, 1);

  // Invoke divergence kernel
  divergenceShared<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_u, d_v, d_w, d_R, d_e, d_f, d_g, d_h, d_m, d_n, alpha1, nx, ny, nz, dx, dy, d_dz_array);


  /////////////////////////////////////////////////
  //                 SOR solver              //////
  /////////////////////////////////////////////////

  int iter = 0;
  std::vector<float> max_error(1, 1.0);

  cudaMalloc((void **)&d_error, 1 * sizeof(float));
  cudaMemcpy(d_error, max_error.data(), 1 * sizeof(float), cudaMemcpyHostToDevice);


  // Main solver loop
  while ((iter < itermax) && (max_error[0] > tol)) {
    // Save previous iteration values for error calculation

    saveLambdaShared<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_lambda, d_lambda_old, nx, ny, nz);
    cudaCheck(cudaGetLastError());
    // cudaMemcpy(d_lambda , lambda.data() , numcell_cent * sizeof(float) , cudaMemcpyHostToDevice);
    int offset = 0;// Red nodes pass
    SOR_RB_Shared<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_lambda, nx, ny, nz, omega, A, B, d_e, d_f, d_g, d_h, d_m, d_n, d_R, offset);
    cudaCheck(cudaGetLastError());

    offset = 1;// Black nodes pass
    SOR_RB_Shared<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_lambda, nx, ny, nz, omega, A, B, d_e, d_f, d_g, d_h, d_m, d_n, d_R, offset);
    cudaCheck(cudaGetLastError());

    dim3 numberOfBlocks2(ceil(domain.numHorizontalCellCentered() / (float)(BLOCKSIZE)), 1, 1);
    // Invoke kernel to apply Neumann boundary condition (lambda (@k=0) = lambda (@k=1))
    applyNeumannBCShared<<<numberOfBlocks2, numberOfThreadsPerBlock>>>(d_lambda, nx, ny);

    calculateErrorShared<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_lambda, d_lambda_old, nx, ny, nz, d_value, d_bvalue, d_error);
    cudaMemcpy(max_error.data(), d_error, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    iter += 1;
  }

  printf("[Solver]\t Residual after %d itertations: %2.9f\n", iter, max_error[0]);
  // std::cout << "Error:" << max_error[0] << "\n";
  // std::cout << "Number of iterations:" << iter << "\n";// Print the number of iterations

  dim3 numberOfBlocks3(ceil(domain.numFaceCentered() / (float)(BLOCKSIZE)), 1, 1);
  // Invoke final velocity (Euler) kernel
  finalVelocityShared<<<numberOfBlocks3, numberOfThreadsPerBlock>>>(d_lambda, d_u, d_v, d_w, d_icellflag, d_f, d_h, d_n, alpha1, alpha2, dx, dy, dz, d_dz_array, nx, ny, nz);
  cudaCheck(cudaGetLastError());

  cudaMemcpy(WGD->u.data(), d_u, numcell_face * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(WGD->v.data(), d_v, numcell_face * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(WGD->w.data(), d_w, numcell_face * sizeof(float), cudaMemcpyDeviceToHost);


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
