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

#ifndef __CUDA_CONCENTRATION_H__
#define __CUDA_CONCENTRATION_H__

#include "plume/Particle.h"

struct ConcentrationParam
{
  float lbndx, lbndy, lbndz;
  float ubndx, ubndy, ubndz;

  float dx, dy, dz;

  int nx, ny, nz;
};

class Concentration
{
public:
  Concentration(const ConcentrationParam param_in) : param(param_in)
  {
    volume = param.dx * param.dy * param.dz;
    numcell = param.nx * param.ny * param.nz;

    h_pBox.resize(numcell, 0.0);
    cudaMalloc(&d_pBox, numcell * sizeof(int));
  }

  ~Concentration()
  {
    cudaFree(d_pBox);
  }

  void collect(const float &timeStep,
               particle_array d_particle_list,
               const int &num_particle);
  void copyback();

  float ongoingAveragingTime = 0.0;
  float volume;


  int *d_pBox;
  std::vector<int> h_pBox;

private:
  const ConcentrationParam param;
  long numcell;
};

#endif
