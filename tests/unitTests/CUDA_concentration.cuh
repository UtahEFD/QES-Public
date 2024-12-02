#pragma once

#include <chrono>
#include <string>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "util/VectorMath.h"
#include "plume/IDGenerator.h"

#include "plume/Particle.h"

#include <cuda.h>
#include <curand.h>

struct ConcentrationParam
{
  float lbndx, lbndy, lbndz;
  float ubndx, ubndy, ubndz;

  float dx, dy, dz;

  int nx, ny, nz;
};

__global__ void collect(int length, particle_array d_particle_list, int *pBox, const ConcentrationParam param);
