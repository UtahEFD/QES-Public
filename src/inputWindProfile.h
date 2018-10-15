#pragma once
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <chrono>

using namespace std;
using std::vector;


void inputWindProfile(float dx, float dy, float dz, int nx, int ny, int nz, double *u0, double *v0, double *w0, int num_sites, int *site_blayer_flag, float *site_one_overL, float *site_xcoord, float *site_ycoord, float *site_wind_dir, float *site_z0, float *site_z_ref, float *site_U_ref, float *x, float *y, float *z);
