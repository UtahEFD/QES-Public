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
 * @file WINDSGeneralData.cu
 * @brief :document this:
 */

#include "WINDSGeneralData.h"

void WINDSGeneralData::allocateDevice()
{
  long numcell_face = domain.numFaceCentered();
  // velocity field components
  cudaMalloc((void **)&d_data.u, numcell_face * sizeof(float));
  cudaMalloc((void **)&d_data.v, numcell_face * sizeof(float));
  cudaMalloc((void **)&d_data.w, numcell_face * sizeof(float));
}
void WINDSGeneralData::freeDevice()
{
  cudaFree(d_data.u);
  cudaFree(d_data.v);
  cudaFree(d_data.w);
}
void WINDSGeneralData::copyDataToDevice()
{
  long numcell_face = domain.numFaceCentered();
  // velocity field components
  cudaMemcpy(d_data.u, u.data(), numcell_face * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_data.v, v.data(), numcell_face * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_data.w, w.data(), numcell_face * sizeof(float), cudaMemcpyHostToDevice);
}
void WINDSGeneralData::copyDataFromDevice()
{
  long numcell_face = domain.numFaceCentered();
  // velocity field components
  cudaMemcpy(u.data(), d_data.u, numcell_face * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(v.data(), d_data.v, numcell_face * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(w.data(), d_data.w, numcell_face * sizeof(float), cudaMemcpyDeviceToHost);
}
