
#pragma once

#include "util/VectorMath.h"
#include "util/VectorMath_CUDA.cuh"

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
  // mat3sym tau_old;
  // mat3sym tau_ddt;

  // vec3 fluxDiv;

} particle;
