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

struct QESTurbData
{
  float *txx;
  float *txy;
  float *txz;
  float *tyy;
  float *tyz;
  float *tzz;

  float *div_tau_x;
  float *div_tau_y;
  float *div_tau_z;

  float *CoEps;
  float *nuT;
};


void test_gpu(const int &, const int &, const int &);
