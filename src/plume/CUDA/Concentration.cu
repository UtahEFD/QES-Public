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
 * This file is part of QES-Plume
 *
 * GPL-3.0 License
 *
 * QES-Plume is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Plume is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Plume. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/** @file
 * @brief
 */

#include "Concentration.h"

__global__ void compute(int length, particle_array d_particle_list, int *pBox, const ConcentrationParam param)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < length) {
    if (d_particle_list.state[idx] == ACTIVE) {
      // x-direction
      int i = floor((d_particle_list.pos[idx]._1 - param.lbndx) / (param.dx + 1e-9));
      // y-direction
      int j = floor((d_particle_list.pos[idx]._2 - param.lbndy) / (param.dy + 1e-9));
      // z-direction
      int k = floor((d_particle_list.pos[idx]._3 - param.lbndz) / (param.dz + 1e-9));

      if (i >= 0 && i <= param.nx - 1 && j >= 0 && j <= param.ny - 1 && k >= 0 && k <= param.nz - 1) {
        int id = k * param.ny * param.nx + j * param.nx + i;
        atomicAdd(&pBox[id], 1);
        // conc[id] = conc[id] + par.m * par.wdecay * timeStep;
      }
    }
  }
}


void Concentration::collect(const float &dt,
                            particle_array d_particle_list,
                            const int &num_particle)
{
  int blockSize = 256;
  int numBlocks = (num_particle + blockSize - 1) / blockSize;

  compute<<<numBlocks, blockSize>>>(num_particle, d_particle_list, d_pBox, param);
  cudaDeviceSynchronize();
  ongoingAveragingTime += dt;
}

void Concentration::copyback()
{
  cudaMemcpy(h_pBox.data(), d_pBox, numcell * sizeof(int), cudaMemcpyDeviceToHost);
}
