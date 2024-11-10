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

/** @file VectorMath_CUDA.h
 * @brief
 */

#ifndef __VECTORMATH_CUDA_H__
#define __VECTORMATH_CUDA_H__

#include <cmath>
#include <cuda.h>
#include "VectorMath.h"

__device__ float length(const vec3 &x);
__device__ float dot(const vec3 &a, const vec3 &b);
__device__ void reflect(const vec3 &n, vec3 &v);
__device__ float distance(const vec3 &a, const vec3 &b);
__device__ void calcInvariants(const mat3sym &tau, vec3 &invar);
__device__ void makeRealizable(const float &invarianceTol, mat3sym &tau);
__device__ bool invert(mat3 &A);
__device__ void multiply(const mat3 &A, const vec3 &b, vec3 &x);

#endif
