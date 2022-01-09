/****************************************************************************
 * Copyright (c) 2021 University of Utah
 * Copyright (c) 2021 University of Minnesota Duluth
 *
 * Copyright (c) 2021 Behnam Bozorgmehr
 * Copyright (c) 2021 Jeremy A. Gibbs
 * Copyright (c) 2021 Fabien Margairaz
 * Copyright (c) 2021 Eric R. Pardyjak
 * Copyright (c) 2021 Zachary Patterson
 * Copyright (c) 2021 Rob Stoll
 * Copyright (c) 2021 Pete Willemsen
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
int index = blockIdx.x*blockDim.x+threadIdx.x;
int stride = blockDim.x*gridDim.x;
for (int it = index; it < icellfluidLength; it+=stride) {
    int cellID = icellfluid2[it];

    // linearized index: cellID = i + j*(nx-1) + k*(nx-1)*(ny-1);
    //  i,j,k -> inverted linearized index
    int k = (int)(cellID / ((nx - 1) * (ny - 1)));
    int j = (int)((cellID - k * (nx - 1) * (ny - 1)) / (nx - 1));
    int i = (int)(cellID - j * (nx - 1) - k * (nx - 1) * (ny - 1));
    int faceID = i + j * nx + k * nx * ny;

    
    // Gxx = dudx
    Gxx[cellID] = (u[faceID + 1] - u[faceID]) / (dx);
    // Gyx = dvdx
    Gyx[cellID] = ((v[faceID + 1] + v[faceID + 1 + nx])
                   - (v[faceID - 1] - v[faceID - 1 + nx]))
                  / (4.0 * dx);
    // Gzx = dwdx
    Gzx[cellID] = ((w[faceID + 1] + w[faceID + 1 + nx * ny]
                    - w[faceID - 1] + w[faceID - 1 + nx * ny]))
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

void TURBGeneralData::getDerivativesGPU(WINDSGeneralData *WGD){

     int gpuID=0;
     cudaError_t errorCheck = cudaGetDevice(&gpuID);

     int smCount=1;
     cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, gpuID);
     
     int threadsPerBlock = 32;
     
     int length = (int)icellfluid.size();

     if(errorCheck==cudaSuccess){
     //temp
     float *Gxx2, *Gxy2, *Gxz2, *Gyx2, *Gyy2, *Gyz2, *Gzx2, *Gzy2, *Gzz2, *WGDu, *WGDv, *WGDw, *WGDx, *WGDy, *WGDz, *WGDdz_array;
     int *icellfluid2;
     
     cudaMalloc((void **)&Gxx2, WGD->numcell_cent * sizeof(float));
     cudaMalloc((void **)&Gxy2, WGD->numcell_cent * sizeof(float));
     cudaMalloc((void **)&Gxz2, WGD->numcell_cent * sizeof(float));
     cudaMalloc((void **)&Gyx2, WGD->numcell_cent * sizeof(float));
     cudaMalloc((void **)&Gyy2, WGD->numcell_cent * sizeof(float));
     cudaMalloc((void **)&Gyz2, WGD->numcell_cent * sizeof(float));
     cudaMalloc((void **)&Gzx2, WGD->numcell_cent * sizeof(float));
     cudaMalloc((void **)&Gzy2, WGD->numcell_cent * sizeof(float));
     cudaMalloc((void **)&Gzz2, WGD->numcell_cent * sizeof(float));
     cudaMalloc((void **)&WGDu, WGD->numcell_face * sizeof(float));
     cudaMalloc((void **)&WGDv, WGD->numcell_face * sizeof(float));
     cudaMalloc((void **)&WGDw, WGD->numcell_face * sizeof(float));
     cudaMalloc((void **)&WGDx, WGD->nx * sizeof(float));
     cudaMalloc((void **)&WGDy, WGD->ny * sizeof(float));
     cudaMalloc((void **)&WGDz, WGD->nz * sizeof(float));
     cudaMalloc((void **)&WGDdz_array, (WGD->nz-1) * sizeof(float));
     cudaMalloc((void **)&icellfluid2, (int)icellfluid.size() * sizeof(int));

     cudaMemcpy(Gxx2, Gxx.data(), WGD->numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(Gxy2, Gxy.data(), WGD->numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(Gxz2, Gxz.data(), WGD->numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(Gyx2, Gyx.data(), WGD->numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(Gyy2, Gyy.data(), WGD->numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(Gyz2, Gyz.data(), WGD->numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(Gzx2, Gzx.data(), WGD->numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(Gzy2, Gzy.data(), WGD->numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(Gzz2, Gzz.data(), WGD->numcell_cent * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(WGDu, WGD->u.data(), WGD->numcell_face * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(WGDv, WGD->v.data(), WGD->numcell_face * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(WGDw, WGD->w.data(), WGD->numcell_face * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(WGDx, WGD->x.data(), WGD->nx * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(WGDy, WGD->y.data(), WGD->ny * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(WGDz, WGD->z.data(), WGD->nz * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(WGDdz_array, WGD->dz_array.data(), (WGD->nz-1) * sizeof(float),  cudaMemcpyHostToDevice);
     cudaMemcpy(icellfluid2, icellfluid.data(), (int)icellfluid.size() * sizeof(int), cudaMemcpyHostToDevice);

//call kernel
     getDerivativesCUDA<<<smCount,threadsPerBlock>>>(WGD->nx, WGD->ny, WGD->nz, WGD->dx, WGD->dy, WGD->dz, Gxx2, Gxy2, Gxz2, Gyx2, Gyy2, Gyz2, Gzx2, Gzy2, Gzz2, flagUniformZGrid, length, WGDu, WGDv, WGDw, WGDx, WGDy, WGDz, WGDdz_array, icellfluid2);

     cudaDeviceSynchronize();

     //cudamemcpy back to host
     cudaMemcpy(Gxx.data(), Gxx2, WGD->numcell_cent * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(Gxy.data(), Gxy2, WGD->numcell_cent * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(Gxz.data(), Gxz2, WGD->numcell_cent * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(Gyx.data(), Gyx2, WGD->numcell_cent * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(Gyy.data(), Gyy2, WGD->numcell_cent * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(Gyz.data(), Gyz2, WGD->numcell_cent * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(Gzx.data(), Gzx2, WGD->numcell_cent * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(Gzy.data(), Gzy2, WGD->numcell_cent * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(Gzz.data(), Gzz2, WGD->numcell_cent * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(WGD->u.data(), WGDu, WGD->numcell_face * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(WGD->v.data(), WGDv, WGD->numcell_face * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(WGD->w.data(), WGDw, WGD->numcell_face * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(WGD->x.data(), WGDx, WGD->nx * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(WGD->y.data(), WGDy, WGD->ny * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(WGD->z.data(), WGDz, WGD->nz * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(WGD->dz_array.data(),WGDdz_array, (WGD->nz-1) * sizeof(float), cudaMemcpyDeviceToHost);
     
     
     //cudafree
     cudaFree(Gxx2);
     cudaFree(Gxy2);
     cudaFree(Gxz2);
     cudaFree(Gyx2);
     cudaFree(Gyy2);
     cudaFree(Gyz2);
     cudaFree(Gzx2);
     cudaFree(Gzy2);
     cudaFree(Gzz2);
     cudaFree(WGDu);
     cudaFree(WGDv);
     cudaFree(WGDw);
     cudaFree(WGDx);
     cudaFree(WGDy);
     cudaFree(WGDz);
     cudaFree(WGDdz_array);
     cudaFree(icellfluid2);
     }
     else{
	printf("CUDA ERROR!\n");
     }
}

