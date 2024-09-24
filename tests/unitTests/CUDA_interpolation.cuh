#pragma once

#include <chrono>
#include <string>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "util/VectorMath.h"
// #include "util/VectorMath_CUDA.cuh"

#include "winds/WINDSGeneralData.h"

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

__global__ void interpolate(int, particle_array, const QESWindsData, const QESgrid &);

__global__ void interpolate(int, particle_array, const WINDSDeviceData, const QESgrid &);

__global__ void interpolate(int, const vec3 *, mat3sym *, vec3 *, const QESTurbData, const QESgrid &q);
__global__ void interpolate(int, particle_array, const QESTurbData, const QESgrid &);

__global__ void interpolate_1(int, particle_array, const QESTurbData, const QESgrid &);
__global__ void interpolate_2(int, particle_array, const QESTurbData, const QESgrid &);
__global__ void interpolate_3(int, particle_array, const QESTurbData, const QESgrid &);
