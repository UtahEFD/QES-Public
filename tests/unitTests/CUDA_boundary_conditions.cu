#include "CUDA_boundary_conditions.cuh"

__device__ bool enforce_exiting(float &pos, const float &domainStart, const float &domainEnd)
{
  if (pos <= domainStart || pos >= domainEnd) {
    return false;
  } else {
    return true;
  }
}

__device__ bool enforce_periodic(float &pos, const float &domainStart, const float &domainEnd)
{
  float domainSize = domainEnd - domainStart;

  if (domainSize != 0) {
    // before beginning of the domain => add domain length
    while (pos < domainStart) {
      pos = pos + domainSize;
    }
    // past end of domain => sub domain length
    while (pos > domainEnd) {
      pos = pos - domainSize;
    }
  }

  return true;
}

__device__ bool enforce_reflection(float &pos, float &velFluct, const float &domainStart, const float &domainEnd)
{

  int reflectCount = 0;
  const int maxReflectCount = 20;

  while ((pos < domainStart || pos > domainEnd) && reflectCount < maxReflectCount) {
    // past end of domain or before beginning of the domain
    if (pos > domainEnd) {
      pos = domainEnd - (pos - domainEnd);
      velFluct = -velFluct;
    } else if (pos < domainStart) {
      pos = domainStart - (pos - domainStart);
      velFluct = -velFluct;
    }
    reflectCount++;
  }// while outside of domain

  // if the velocity is so large that the particle would reflect more than 100
  // times, the boundary condition could fail.
  if (reflectCount == maxReflectCount) {
    return false;
    /*if (pos > domainEnd) {
      return false;
    } else if (pos < domainStart) {
      return false;
      }*/
  } else {
    return true;
  }
}

__device__ void boundary_conditions(particle_array p, int idx, const BC_Params &bc_param)
{
  vec3 pos = p.pos[idx];
  if (!enforce_exiting(pos._1, bc_param.xStartDomain, bc_param.xEndDomain)) {
    p.state[idx] = INACTIVE;
  }
  if (!enforce_exiting(pos._2, bc_param.yStartDomain, bc_param.yEndDomain)) {
    p.state[idx] = INACTIVE;
  }
  if (!enforce_exiting(pos._3, bc_param.zStartDomain, bc_param.zEndDomain)) {
    p.state[idx] = INACTIVE;
  }
}

// test boundary conditon as kernel vs device function
__global__ void boundary_conditions(int length, particle_array p, const BC_Params &bc_param)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    vec3 pos = p.pos[idx];
    if (!enforce_exiting(pos._1, bc_param.xStartDomain, bc_param.xEndDomain)) {
      p.state[idx] = INACTIVE;
    }
    if (!enforce_exiting(pos._2, bc_param.yStartDomain, bc_param.yEndDomain)) {
      p.state[idx] = INACTIVE;
    }
    if (!enforce_exiting(pos._3, bc_param.zStartDomain, bc_param.zEndDomain)) {
      p.state[idx] = INACTIVE;
    }
  }
}
