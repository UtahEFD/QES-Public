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
__global__ void getDerivatives_v2_CUDA(WINDSGeneralData *WGD, int *icellfluid, int nx, int ny, int nz, float *Gxx, float *Gxy, float *Gxz, float *Gyx, float *Gyy, float *Gyz, float *Gzx, float *Gzy, float *Gzz, bool flagUniformZGrid, int icellfluidLength, float *u, float *v, float *w, float *x, float *y, float *z, float *dz_array)
{
//////
//////NEED TO REFACTOR EVERYTHING TO NOT USE std::vector
//////
  for (int *it = icellfluid; it <= (icellfluid+icellfluidLength); ++it) {
    int cellID = *it;

    // linearized index: cellID = i + j*(nx-1) + k*(nx-1)*(ny-1);
    //  i,j,k -> inverted linearized index
    int k = (int)(cellID / ((nx - 1) * (ny - 1)));
    int j = (int)((cellID - k * (nx - 1) * (ny - 1)) / (nx - 1));
    int i = cellID - j * (nx - 1) - k * (nx - 1) * (ny - 1);
    int faceID = i + j * nx + k * nx * ny;

    /*
     - Diagonal componants of the velocity gradient tensor naturally fall at the cell-center
     - Off-diagonal componants of the  velocity gradient tensor require extra interpolation
       of the velocity field to get the derivative at the cell-center  
     - Derivative with respect to z need to be adjusted for non-uniform z-grid
    */

    // Gxx = dudx
    Gxx[cellID] = (u[faceID + 1] - u[faceID]) / (WGD->dx);
    // Gyx = dvdx
    Gyx[cellID] = ((v[faceID + 1] + v[faceID + 1 + nx])
                   - (v[faceID - 1] - v[faceID - 1 + nx]))
                  / (4.0 * WGD->dx);
    // Gzx = dwdx
    Gzx[cellID] = ((w[faceID + 1] + w[faceID + 1 + nx * ny]
                    - w[faceID - 1] + w[faceID - 1 + nx * ny]))
                  / (4.0 * WGD->dx);

    // Gxy = dudy
    Gxy[cellID] = ((u[faceID + nx] + u[faceID + 1 + nx])
                   - (u[faceID - nx] + u[faceID + 1 - nx]))
                  / (4.0 * WGD->dy);
    // Gyy = dvdy
    Gyy[cellID] = (v[faceID + nx] - v[faceID]) / (WGD->dy);
    // Gzy = dwdy
    Gzy[cellID] = ((w[faceID + nx] + w[faceID + nx + nx * ny])
                   - (w[faceID - nx] + w[faceID - nx + nx * ny]))
                  / (4.0 * WGD->dy);


    if (flagUniformZGrid) {
      // Gxz = dudz
      Gxz[cellID] = ((u[faceID + nx * ny] + u[faceID + 1 + nx * ny])
                     - (u[faceID - nx * ny] + u[faceID + 1 - nx * ny]))
                    / (4.0 * WGD->dz);
      // Gyz = dvdz
      Gyz[cellID] = ((v[faceID + nx * ny] + v[faceID + nx + nx * ny])
                     - (v[faceID - nx * ny] + v[faceID + nx - nx * ny]))
                    / (4.0 * WGD->dz);
      // Gzz = dwdz
      Gzz[cellID] = (w[faceID + nx * ny] - w[faceID]) / (WGD->dz);
    } else {
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

void TURBGeneralData::getDerivatives_v2(WINDSGeneralData *WGD){
     int np_cc = (nz-1) * (ny-1) * (nx-1);
     int *icellfluid2;
     int icellfluidLength;

     //temp
     float *Gxx2, *Gxy2, *Gxz2, *Gyx2, *Gyy2, *Gyz2, *Gzx2, *Gzy2, *Gzz2, *WGDu, *WGDv, *WGDw, *WGDx, *WGDy, *WGDz, *WGDdz_array;
     
     //cudaMalloc(&icellfluidLength, sizeof(int));
     cudaMalloc((void **)&icellfluid2, WGD->numcell_cent * sizeof(int));
     cudaMalloc((void **)&Gxx2, np_cc * sizeof(float));
     cudaMalloc((void **)&Gxy2, np_cc * sizeof(float));
     cudaMalloc((void **)&Gxz2, np_cc * sizeof(float));
     cudaMalloc((void **)&Gyx2, np_cc * sizeof(float));
     cudaMalloc((void **)&Gyy2, np_cc * sizeof(float));
     cudaMalloc((void **)&Gyz2, np_cc * sizeof(float));
     cudaMalloc((void **)&Gzx2, np_cc * sizeof(float));
     cudaMalloc((void **)&Gzy2, np_cc * sizeof(float));
     cudaMalloc((void **)&Gzz2, np_cc * sizeof(float));
     cudaMalloc((void **)&WGDu, WGD->numcell_face * sizeof(float));
     cudaMalloc((void **)&WGDv, WGD->numcell_face * sizeof(float));
     cudaMalloc((void **)&WGDw, WGD->numcell_face * sizeof(float));
     cudaMalloc((void **)&WGDx, WGD->nx * sizeof(float));
     cudaMalloc((void **)&WGDy, WGD->ny * sizeof(float));
     cudaMalloc((void **)&WGDz, WGD->nz * sizeof(float));
     cudaMalloc((void **)&WGDdz_array, (WGD->nz-1) * sizeof(float));

     //cudamemcpy to gpu

     //cudaMemcpy(icellfluidLength, icellfluid.size(), sizeof(int), cudaMemcpyHostToDevice);
     cudaMemcpy(icellfluid2, icellfluid.data(), WGD->numcell_cent * sizeof(int), cudaMemcpyHostToDevice);
     cudaMemcpy(Gxx2, Gxx.data(), np_cc * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(Gxy2, Gxy.data(), np_cc * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(Gxz2, Gxz.data(), np_cc * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(Gyx2, Gyx.data(), np_cc * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(Gyy2, Gyy.data(), np_cc * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(Gyz2, Gyz.data(), np_cc * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(Gzx2, Gzx.data(), np_cc * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(Gzy2, Gzy.data(), np_cc * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(Gzz2, Gzz.data(), np_cc * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(WGDu, WGD->u.data(), WGD->numcell_face * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(WGDv, WGD->v.data(), WGD->numcell_face * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(WGDw, WGD->w.data(), WGD->numcell_face * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(WGDx, WGD->x.data(), WGD->nx * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(WGDy, WGD->y.data(), WGD->ny * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(WGDz, WGD->z.data(), WGD->nz * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(WGDdz_array, WGD->dz_array.data(), (WGD->nz-1) * sizeof(float),  cudaMemcpyHostToDevice);

//call kernel
     getDerivatives_v2_CUDA<<<1,1>>>(WGD, icellfluid2, nx, ny, nz, Gxx2, Gxy2, Gxz2, Gyx2, Gyy2, Gyz2, Gzx2, Gzy2, Gzz2, flagUniformZGrid, icellfluidLength, WGDu, WGDv, WGDw, WGDx, WGDy, WGDz, WGDdz_array);

     //cudamemcpy back to host
     cudaMemcpy(icellfluid.data(), icellfluid2, WGD->numcell_cent * sizeof(int), cudaMemcpyDeviceToHost);
     cudaMemcpy(Gxx.data(), Gxx2, np_cc * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(Gxy.data(), Gxy2, np_cc * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(Gxz.data(), Gxz2, np_cc * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(Gyx.data(), Gyx2, np_cc * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(Gyy.data(), Gyy2, np_cc * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(Gyz.data(), Gyz2, np_cc * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(Gzx.data(), Gzx2, np_cc * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(Gzy.data(), Gzy2, np_cc * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(Gzz.data(), Gzz2, np_cc * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(WGD->u.data(), WGDu, WGD->numcell_face * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(WGD->v.data(), WGDv, WGD->numcell_face * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(WGD->w.data(), WGDw, WGD->numcell_face * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(WGD->x.data(), WGDx, WGD->nx * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(WGD->y.data(), WGDy, WGD->ny * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(WGD->z.data(), WGDz, WGD->nz * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(WGD->dz_array.data(),WGDdz_array, (WGD->nz-1) * sizeof(float), cudaMemcpyDeviceToHost);
     
     //cudafree
     //cudaFree(icellfluidLength);
     cudaFree(icellfluid2);
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
}

