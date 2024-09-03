#pragma once

#include <chrono>
#include <string>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "util/VectorMath.h"
// #include "util/VectorMath_CUDA.cuh"

#include <cuda.h>
#include <curand.h>

#include "Particle.h"
#include "CUDA_QES_Data.h"

struct interpWeight
{
  int ii;// nearest cell index to the left in the x direction
  int jj;// nearest cell index to the left in the y direction
  int kk;// nearest cell index to the left in the z direction
  float iw;// normalized distance to the nearest cell index to the left in the x direction
  float jw;// normalized distance to the nearest cell index to the left in the y direction
  float kw;// normalized distance to the nearest cell index to the left in the z direction
};

__global__ void interpolate(int length, particle_array d_particle_list, const QESWindsData data, const QESgrid &qes_grid);
__global__ void interpolate(int length, const vec3 *pos, mat3sym *tau, vec3 *sigma, const QESTurbData data, const QESgrid &qes_grid);
__global__ void interpolate(int length, particle_array d_particle_list, const QESTurbData data, const QESgrid &qes_grid);
