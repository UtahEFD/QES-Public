#pragma once

#include <chrono>
#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "util/VectorMath.h"

#include <cuda.h>
#include <curand.h>

enum state : int { ACTIVE = 0,
                   INACTIVE = 1,
                   ROGUE = 2 };

typedef struct
{
  bool isRogue;
  bool isActive;
  uint32_t ID;

  float CoEps;

  vec3 pos;

  vec3 velMean;

  vec3 velFluct;
  vec3 velFluct_old;
  vec3 delta_velFluct;

  mat3sym tau;
  mat3sym tau_old;

  vec3 fluxDiv;

} particle;

typedef struct
{
  int *state;
  uint32_t *ID;

  float *CoEps;

  vec3 *pos;

  vec3 *velMean;

  vec3 *velFluct;
  vec3 *velFluct_old;
  vec3 *delta_velFluct;

  mat3sym *tau;
  mat3sym *tau_old;

  vec3 *flux_div;

} particle_array;


void test_gpu(const int &, const int &, const int &);
