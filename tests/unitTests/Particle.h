#pragma once

#include "util/VectorMath.h"


typedef struct
{
  int state;
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
