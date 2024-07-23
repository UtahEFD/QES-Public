#pragma once

#include <chrono>
#include <string>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "util/VectorMath.h"
#include "Particle.h"

#include <cuda.h>
#include <curand.h>

struct QESgrid
{
  float dx;
  float dy;
  float dz;

  int nx;
  int ny;
  int nz;
};

struct QESWindsData
{
  float *u;
  float *v;
  float *w;
};

void test_gpu(const int &, const int &, const int &);
