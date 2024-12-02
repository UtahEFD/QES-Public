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

#pragma once

#include <vector>

#include <cuda.h>
#include <curand.h>

#include "util/VectorMath.h"

#include "plume/Particle.h"
#include "plume/cuda/QES_data.h"

struct interpWeight
{
  int ii;// nearest cell index to the left in the x direction
  int jj;// nearest cell index to the left in the y direction
  int kk;// nearest cell index to the left in the z direction
  float iw;// normalized distance to the nearest cell index to the left in the x direction
  float jw;// normalized distance to the nearest cell index to the left in the y direction
  float kw;// normalized distance to the nearest cell index to the left in the z direction
};

class Interpolation
{
public:
  Interpolation()
  {
  }

  ~Interpolation()
  {
  }

  void get(particle_array, const QESWindsData &, const QESTurbData &, const QESgrid &, const int &);

  void get(particle_array, const QESTurbData &, const QESgrid &, const int &);

private:
};
