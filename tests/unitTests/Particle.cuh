
#pragma once

#include "util/VectorMath.h"
#include "util/VectorMath_CUDA.cuh"

enum state : int { ACTIVE = 0,
                   INACTIVE = 1,
                   ROGUE = 2 };

typedef struct
{
  bool isRogue;
  bool isActive;

  float CoEps;

  vec3 pos;

  vec3 velMean;

  vec3 velFluct;
  vec3 velFluct_old;
  vec3 delta_velFluct;

  mat3sym tau;
  mat3sym tau_old;

  vec3 fluxDiv;

} particle_AOS;

typedef struct
{
  int *state;

  float *CoEps;

  vec3 *pos;

  vec3 *velMean;

  vec3 *velFluct;
  vec3 *velFluct_old;
  vec3 *delta_velFluct;

  mat3sym *tau;
  mat3sym *tau_old;

  vec3 *flux_div;

} particle_SOA;
