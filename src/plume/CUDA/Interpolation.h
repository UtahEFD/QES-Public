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

#ifndef __CUDA_INTERPOLATION_H__
#define __CUDA_INTERPOLATION_H__

#include <vector>

#include <cuda.h>
#include <curand.h>

#include "util/VectorMath.h"

#include "plume/Particle.h"
#include "plume/CUDA/QES_data.h"

struct interpWeight
{
  int ii;// nearest cell index to the left in the x direction
  int jj;// nearest cell index to the left in the y direction
  int kk;// nearest cell index to the left in the z direction
  float iw;// normalized distance to the nearest cell index to the left in the x direction
  float jw;// normalized distance to the nearest cell index to the left in the y direction
  float kw;// normalized distance to the nearest cell index to the left in the z direction
};

__device__ void setInterp3Dindex_uFace(vec3 &pos, interpWeight &wgt, const QESgrid qes_grid);
__device__ void setInterp3Dindex_vFace(vec3 &pos, interpWeight &wgt, const QESgrid qes_grid);
__device__ void setInterp3Dindex_wFace(vec3 &pos, interpWeight &wgt, const QESgrid qes_grid);
__device__ void interp3D_faceVar(float &out, const float *data, const interpWeight &wgt, const QESgrid qes_grid);

__device__ void setInterp3Dindex_cellVar(const vec3 &pos, interpWeight &wgt, const QESgrid qes_grid);
__device__ void interp3D_cellVar(float &out, const float *data, const interpWeight &wgt, const QESgrid qes_grid);

__global__ void interpolate(int length, particle_array d_particle_list, const QESWindsData data, const QESgrid qes_grid);
__global__ void interpolate(int length, particle_array d_particle_list, const WINDSDeviceData data, const QESgrid qes_grid);
__global__ void interpolate(int length, const vec3 *pos, mat3sym *tau, vec3 *sigma, const QESTurbData data, const QESgrid qes_grid);
__global__ void interpolate(int length, particle_array d_particle_list, const QESTurbData data, const QESgrid qes_grid);

__global__ void interpolate_1(int length, particle_array d_particle_list, const QESTurbData data, const QESgrid qes_grid);
__global__ void interpolate_2(int length, particle_array d_particle_list, const QESTurbData data, const QESgrid qes_grid);
__global__ void interpolate_3(int length, particle_array d_particle_list, const QESTurbData data, const QESgrid qes_grid);


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

#endif
