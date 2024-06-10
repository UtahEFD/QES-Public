
#pragma once
#include "util/VectorMath_CUDA.h"

typedef struct
{
  bool isRogue;
  bool isActive;
  vec3 pos;

  vec3 uMean;
  vec3 uFluct;
  vec3 uFluct_old;
  vec3 uFluct_delta;

  mat3sym tau;
  mat3sym tau_old;
  mat3sym tau_ddt;

  vec3 fluxDiv;

} particle;