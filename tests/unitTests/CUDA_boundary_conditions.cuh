#pragma once

#include <chrono>
#include <string>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "util/VectorMath.h"
// #include "util/VectorMath_CUDA.cuh"

#include "plume/Particle.h"

#include <cuda.h>
#include <curand.h>

typedef struct
{
  float xStartDomain;
  float yStartDomain;
  float zStartDomain;

  float xEndDomain;
  float yEndDomain;
  float zEndDomain;

} BC_Params;

__device__ void boundary_conditions(particle_array p, int idx, const BC_Params &bc_param);

// test boundary conditon as kernel vs device function
__global__ void boundary_conditions(int length, particle_array p, const BC_Params &bc_param);
