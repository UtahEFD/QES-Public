#pragma once

#include <chrono>
#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "util/VectorMath.h"

#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"

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
  float *tke;
};

void copy_data_gpu(const WINDSGeneralData *WGD, QESWindsData &d_qes_winds_data);
void copy_data_gpu(const TURBGeneralData *TGD, QESTurbData &d_qes_turb_data);
void copy_data_gpu(const int &num_face, QESWindsData &d_qes_winds_data);
void copy_data_gpu(const int &num_cell, QESTurbData &d_qes_turb_data);
