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
 * @file TURBGeneralData.cu
 * @brief :document this:
 */

#include "TURBGeneralData.h"
__global__ void getDerivativesCUDA(int nx, int ny, int nz, float dx, float dy, float dz, float *Gxx, float *Gxy, float *Gxz, float *Gyx, float *Gyy, float *Gyz, float *Gzx, float *Gzy, float *Gzz, bool flagUniformZGrid, int icellfluidLength, float *u, float *v, float *w, float *x, float *y, float *z, float *dz_array, int *icellfluid2)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int it = index; it < icellfluidLength; it += stride) {
    int cellID = icellfluid2[it];
    // int cellID = it;
    //  linearized index: cellID = i + j*(nx-1) + k*(nx-1)*(ny-1);
    //   i,j,k -> inverted linearized index
    int k = (int)(cellID / ((nx - 1) * (ny - 1)));
    int j = (int)((cellID - k * (nx - 1) * (ny - 1)) / (nx - 1));
    int i = (int)(cellID - j * (nx - 1) - k * (nx - 1) * (ny - 1));
    int faceID = i + j * nx + k * nx * ny;


    // Gxx = dudx
    Gxx[cellID] = (u[faceID + 1] - u[faceID]) / (dx);
    // Gyx = dvdx
    Gyx[cellID] = ((v[faceID + 1] + v[faceID + 1 + nx])
                   - (v[faceID - 1] + v[faceID - 1 + nx]))
                  / (4.0 * dx);
    // Gzx = dwdx
    Gzx[cellID] = ((w[faceID + 1] + w[faceID + 1 + nx * ny])
                   - (w[faceID - 1] + w[faceID - 1 + nx * ny]))
                  / (4.0 * dx);

    // Gxy = dudy
    Gxy[cellID] = ((u[faceID + nx] + u[faceID + 1 + nx])
                   - (u[faceID - nx] + u[faceID + 1 - nx]))
                  / (4.0 * dy);
    // Gyy = dvdy
    Gyy[cellID] = (v[faceID + nx] - v[faceID]) / (dy);
    // Gzy = dwdy
    Gzy[cellID] = ((w[faceID + nx] + w[faceID + nx + nx * ny])
                   - (w[faceID - nx] + w[faceID - nx + nx * ny]))
                  / (4.0 * dy);


    if (flagUniformZGrid) {
      // Gxz = dudz
      Gxz[cellID] = ((u[faceID + nx * ny] + u[faceID + 1 + nx * ny])
                     - (u[faceID - nx * ny] + u[faceID + 1 - nx * ny]))
                    / (4.0 * dz);
      // Gyz = dvdz
      Gyz[cellID] = ((v[faceID + nx * ny] + v[faceID + nx + nx * ny])
                     - (v[faceID - nx * ny] + v[faceID + nx - nx * ny]))
                    / (4.0 * dz);
      // Gzz = dwdz
      Gzz[cellID] = (w[faceID + nx * ny] - w[faceID]) / (dz);
    }

    else {
      // Gxz = dudz
      Gxz[cellID] = (0.5 * (z[k] - z[k - 1]) / (z[k + 1] - z[k])
                       * ((u[faceID + nx * ny] + u[faceID + 1 + nx * ny])
                          - (u[faceID] + u[faceID + 1]))
                     + 0.5 * (z[k + 1] - z[k]) / (z[k] - z[k - 1])
                         * ((u[faceID] + u[faceID + 1])
                            - (u[faceID - nx * ny] + u[faceID + 1 - nx * ny])))
                    / (z[k + 1] - z[k - 1]);
      // Gyz = dvdz
      Gyz[cellID] = (0.5 * (z[k] - z[k - 1]) / (z[k + 1] - z[k])
                       * ((v[faceID + nx * ny] + v[faceID + nx + nx * ny])
                          - (v[faceID] + v[faceID + nx]))
                     + 0.5 * (z[k + 1] - z[k]) / (z[k] - z[k - 1])
                         * ((v[faceID] + v[faceID + nx])
                            - (v[faceID - nx * ny] + v[faceID + nx - nx * ny])))
                    / (z[k + 1] - z[k - 1]);
      // Gzz = dwdz
      Gzz[cellID] = (w[faceID + nx * ny] - w[faceID]) / (dz_array[k]);
    }
  }
  return;
}

__global__ void uDerivatives(int nx, int ny, int nz, float dx, float dy, float dz, float *Gxx, float *Gxy, float *Gxz, bool flagUniformZGrid, int icellfluidLength, float *u, float *x, float *y, float *z, float *dz_array, int *icellfluid2)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int it = index; it < icellfluidLength; it += stride) {
    int cellID = icellfluid2[it];
    // int cellID = it;
    //  linearized index: cellID = i + j*(nx-1) + k*(nx-1)*(ny-1);
    //   i,j,k -> inverted linearized index
    int k = (int)(cellID / ((nx - 1) * (ny - 1)));
    int j = (int)((cellID - k * (nx - 1) * (ny - 1)) / (nx - 1));
    int i = (int)(cellID - j * (nx - 1) - k * (nx - 1) * (ny - 1));
    int faceID = i + j * nx + k * nx * ny;


    // Gxx = dudx
    Gxx[cellID] = (u[faceID + 1] - u[faceID]) / (dx);

    // Gxy = dudy
    Gxy[cellID] = ((u[faceID + nx] + u[faceID + 1 + nx])
                   - (u[faceID - nx] + u[faceID + 1 - nx]))
                  / (4.0 * dy);

    if (flagUniformZGrid) {
      // Gxz = dudz
      Gxz[cellID] = ((u[faceID + nx * ny] + u[faceID + 1 + nx * ny])
                     - (u[faceID - nx * ny] + u[faceID + 1 - nx * ny]))
                    / (4.0 * dz);
    } else {
      // Gxz = dudz
      Gxz[cellID] = (0.5 * (z[k] - z[k - 1]) / (z[k + 1] - z[k])
                       * ((u[faceID + nx * ny] + u[faceID + 1 + nx * ny])
                          - (u[faceID] + u[faceID + 1]))
                     + 0.5 * (z[k + 1] - z[k]) / (z[k] - z[k - 1])
                         * ((u[faceID] + u[faceID + 1])
                            - (u[faceID - nx * ny] + u[faceID + 1 - nx * ny])))
                    / (z[k + 1] - z[k - 1]);
    }
  }
  return;
}

__global__ void vDerivatives(int nx, int ny, int nz, float dx, float dy, float dz, float *Gyx, float *Gyy, float *Gyz, bool flagUniformZGrid, int icellfluidLength, float *v, float *x, float *y, float *z, float *dz_array, int *icellfluid2)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int it = index; it < icellfluidLength; it += stride) {
    int cellID = icellfluid2[it];
    // int cellID = it;
    //  linearized index: cellID = i + j*(nx-1) + k*(nx-1)*(ny-1);
    //   i,j,k -> inverted linearized index
    int k = (int)(cellID / ((nx - 1) * (ny - 1)));
    int j = (int)((cellID - k * (nx - 1) * (ny - 1)) / (nx - 1));
    int i = (int)(cellID - j * (nx - 1) - k * (nx - 1) * (ny - 1));
    int faceID = i + j * nx + k * nx * ny;

    // Gyx = dvdx
    Gyx[cellID] = ((v[faceID + 1] + v[faceID + 1 + nx])
                   - (v[faceID - 1] + v[faceID - 1 + nx]))
                  / (4.0 * dx);

    // Gyy = dvdy
    Gyy[cellID] = (v[faceID + nx] - v[faceID]) / (dy);


    if (flagUniformZGrid) {
      // Gyz = dvdz
      Gyz[cellID] = ((v[faceID + nx * ny] + v[faceID + nx + nx * ny])
                     - (v[faceID - nx * ny] + v[faceID + nx - nx * ny]))
                    / (4.0 * dz);
    } else {
      // Gyz = dvdz
      Gyz[cellID] = (0.5 * (z[k] - z[k - 1]) / (z[k + 1] - z[k])
                       * ((v[faceID + nx * ny] + v[faceID + nx + nx * ny])
                          - (v[faceID] + v[faceID + nx]))
                     + 0.5 * (z[k + 1] - z[k]) / (z[k] - z[k - 1])
                         * ((v[faceID] + v[faceID + nx])
                            - (v[faceID - nx * ny] + v[faceID + nx - nx * ny])))
                    / (z[k + 1] - z[k - 1]);
    }
  }
  return;
}

__global__ void wDerivatives(int nx, int ny, int nz, float dx, float dy, float dz, float *Gzx, float *Gzy, float *Gzz, bool flagUniformZGrid, int icellfluidLength, float *w, float *x, float *y, float *z, float *dz_array, int *icellfluid2)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int it = index; it < icellfluidLength; it += stride) {
    int cellID = icellfluid2[it];
    // int cellID = it;
    //  linearized index: cellID = i + j*(nx-1) + k*(nx-1)*(ny-1);
    //   i,j,k -> inverted linearized index
    int k = (int)(cellID / ((nx - 1) * (ny - 1)));
    int j = (int)((cellID - k * (nx - 1) * (ny - 1)) / (nx - 1));
    int i = (int)(cellID - j * (nx - 1) - k * (nx - 1) * (ny - 1));
    int faceID = i + j * nx + k * nx * ny;

    // Gzx = dwdx
    Gzx[cellID] = ((w[faceID + 1] + w[faceID + 1 + nx * ny])
                   - (w[faceID - 1] + w[faceID - 1 + nx * ny]))
                  / (4.0 * dx);

    // Gzy = dwdy
    Gzy[cellID] = ((w[faceID + nx] + w[faceID + nx + nx * ny])
                   - (w[faceID - nx] + w[faceID - nx + nx * ny]))
                  / (4.0 * dy);


    if (flagUniformZGrid) {
      // Gzz = dwdz
      Gzz[cellID] = (w[faceID + nx * ny] - w[faceID]) / (dz);
    } else {
      // Gzz = dwdz
      Gzz[cellID] = (w[faceID + nx * ny] - w[faceID]) / (dz_array[k]);
    }
  }
  return;
}

void TURBGeneralData::getDerivativesGPU()
{
  auto [nx, ny, nz] = domain.getDomainCellNum();
  auto [dx, dy, dz] = domain.getDomainSize();
  long numcell_face = domain.numFaceCentered();
  long numcell_cent = domain.numCellCentered();

  int gpuID = 0;
  cudaError_t errorCheck = cudaGetDevice(&gpuID);

  int blockCount = 1;
  cudaDeviceGetAttribute(&blockCount, cudaDevAttrMultiProcessorCount, gpuID);
  std::cout << blockCount << std::endl;

  int threadsPerBlock = 256;
  // cudaDeviceGetAttribute(&threadsPerBlock, cudaDevAttrMaxThreadsPerBlock, gpuID);
  std::cout << threadsPerBlock << std::endl;

  int length = (int)icellfluid.size();

  int blockSize = 1024;
  dim3 numberOfThreadsPerBlock(blockSize, 1, 1);
  dim3 numberOfBlocks(ceil(length / (float)(blockSize)), 1, 1);

  if (errorCheck == cudaSuccess) {
    // temp
    float *d_Gxx, *d_Gxy, *d_Gxz, *d_Gyx, *d_Gyy, *d_Gyz, *d_Gzx, *d_Gzy, *d_Gzz;
    float *d_u, *d_v, *d_w, *d_x, *d_y, *d_z, *d_dz_array;
    int *d_icellfluid;

    cudaMalloc((void **)&d_Gxx, numcell_cent * sizeof(float));
    cudaMalloc((void **)&d_Gxy, numcell_cent * sizeof(float));
    cudaMalloc((void **)&d_Gxz, numcell_cent * sizeof(float));
    cudaMalloc((void **)&d_Gyx, numcell_cent * sizeof(float));
    cudaMalloc((void **)&d_Gyy, numcell_cent * sizeof(float));
    cudaMalloc((void **)&d_Gyz, numcell_cent * sizeof(float));
    cudaMalloc((void **)&d_Gzx, numcell_cent * sizeof(float));
    cudaMalloc((void **)&d_Gzy, numcell_cent * sizeof(float));
    cudaMalloc((void **)&d_Gzz, numcell_cent * sizeof(float));
    cudaMalloc((void **)&d_u, numcell_face * sizeof(float));
    cudaMalloc((void **)&d_v, numcell_face * sizeof(float));
    cudaMalloc((void **)&d_w, numcell_face * sizeof(float));
    cudaMalloc((void **)&d_x, (nx - 1) * sizeof(float));
    cudaMalloc((void **)&d_y, (ny - 1) * sizeof(float));
    cudaMalloc((void **)&d_z, (nz - 1) * sizeof(float));
    cudaMalloc((void **)&d_dz_array, (nz - 1) * sizeof(float));
    cudaMalloc((void **)&d_icellfluid, (int)icellfluid.size() * sizeof(int));

    cudaMemcpy(d_u, m_WGD->u.data(), numcell_face * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, m_WGD->v.data(), numcell_face * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, m_WGD->w.data(), numcell_face * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, domain.x.data(), (nx - 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, domain.y.data(), (ny - 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, domain.z.data(), (nz - 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dz_array, domain.dz_array.data(), (nz - 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_icellfluid, icellfluid.data(), (int)icellfluid.size() * sizeof(int), cudaMemcpyHostToDevice);

    // call kernel

    auto gpuStartTime = std::chrono::high_resolution_clock::now();

    getDerivativesCUDA<<<blockCount, threadsPerBlock>>>(nx, ny, nz, dx, dy, dz, d_Gxx, d_Gxy, d_Gxz, d_Gyx, d_Gyy, d_Gyz, d_Gzx, d_Gzy, d_Gzz, flagUniformZGrid, length, d_u, d_v, d_w, d_x, d_y, d_z, d_dz_array, d_icellfluid);
    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) std::cout << "CUDA KERNEL ERROR: " << cudaGetErrorString(kernelError) << "\n";
    cudaDeviceSynchronize();

    auto gpuEndTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpuElapsed = gpuEndTime - gpuStartTime;
    std::cout << "\t\t GPU Derivatives: elapsed time: " << gpuElapsed.count() << " s\n";

    gpuStartTime = std::chrono::high_resolution_clock::now();

    uDerivatives<<<blockCount, threadsPerBlock>>>(nx, ny, nz, dx, dy, dz, d_Gxx, d_Gxy, d_Gxz, flagUniformZGrid, length, d_u, d_x, d_y, d_z, d_dz_array, d_icellfluid);
    vDerivatives<<<blockCount, threadsPerBlock>>>(nx, ny, nz, dx, dy, dz, d_Gyx, d_Gyy, d_Gyz, flagUniformZGrid, length, d_v, d_x, d_y, d_z, d_dz_array, d_icellfluid);
    wDerivatives<<<blockCount, threadsPerBlock>>>(nx, ny, nz, dx, dy, dz, d_Gzx, d_Gzy, d_Gzz, flagUniformZGrid, length, d_w, d_x, d_y, d_z, d_dz_array, d_icellfluid);
    kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) std::cout << "CUDA KERNEL ERROR: " << cudaGetErrorString(kernelError) << "\n";
    cudaDeviceSynchronize();

    gpuEndTime = std::chrono::high_resolution_clock::now();
    gpuElapsed = gpuEndTime - gpuStartTime;
    std::cout << "\t\t GPU Derivatives: elapsed time: " << gpuElapsed.count() << " s\n";

    /*
    gpuStartTime = std::chrono::high_resolution_clock::now();
    getDerivativesCUDA<<<100, 100>>>(WGD->nx, WGD->ny, WGD->nz, WGD->dx, WGD->dy, WGD->dz, d_Gxx, d_Gxy, d_Gxz, d_Gyx, d_Gyy, d_Gyz, d_Gzx, d_Gzy, d_Gzz, flagUniformZGrid, length, d_u, d_v, d_w, d_x, d_y, d_z, WGDdz_array, icellfluid2);
    cudaDeviceSynchronize();

    gpuEndTime = std::chrono::high_resolution_clock::now();
    gpuElapsed = gpuEndTime - gpuStartTime;
    std::cout << "\t\t GPU Derivatives: elapsed time: " << gpuElapsed.count() << " s\n";
*/
    // cudamemcpy back to host
    cudaMemcpy(Gxx.data(), d_Gxx, numcell_cent * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Gxy.data(), d_Gxy, numcell_cent * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Gxz.data(), d_Gxz, numcell_cent * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Gyx.data(), d_Gyx, numcell_cent * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Gyy.data(), d_Gyy, numcell_cent * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Gyz.data(), d_Gyz, numcell_cent * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Gzx.data(), d_Gzx, numcell_cent * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Gzy.data(), d_Gzy, numcell_cent * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Gzz.data(), d_Gzz, numcell_cent * sizeof(float), cudaMemcpyDeviceToHost);

    // cudafree
    cudaFree(d_Gxx);
    cudaFree(d_Gxy);
    cudaFree(d_Gxz);
    cudaFree(d_Gyx);
    cudaFree(d_Gyy);
    cudaFree(d_Gyz);
    cudaFree(d_Gzx);
    cudaFree(d_Gzy);
    cudaFree(d_Gzz);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_dz_array);
    cudaFree(d_icellfluid);
  } else {
    std::cout << "CUDA ERROR: " << cudaGetErrorString(cudaGetLastError()) << "\n";
  }
}
