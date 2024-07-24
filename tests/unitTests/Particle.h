#pragma once

#include "util/VectorMath.h"

enum ParticleStates : int { ACTIVE,
                            INACTIVE,
                            ROGUE };

typedef struct
{
  float xStartDomain;
  float yStartDomain;
  float zStartDomain;

  float xEndDomain;
  float yEndDomain;
  float zEndDomain;

} BC_Params;

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

typedef struct
{
  int *state;
  uint32_t *ID;

  float *CoEps;
  float *nuT;

  vec3 *pos;

  vec3 *velMean;

  vec3 *velFluct;
  vec3 *velFluct_old;
  vec3 *delta_velFluct;

  mat3sym *tau;
  mat3sym *tau_old;

  vec3 *flux_div;

} particle_array;
