#include "Model.h"

#include "util/VectorMath_CUDA.cuh"

__device__ void solve(particle_array p, int tid, float par_dt, float invarianceTol, float vel_threshold, vec3 vRandn)
{

  float CoEps = p.CoEps[tid];
  // bool isActive;
  // bool isRogue;


  // now need to call makeRealizable on tau
  makeRealizable(invarianceTol, p.tau[tid]);

  // now need to calculate the inverse values for tau
  // directly modifies the values of tau
  mat3 L = { p.tau[tid]._11, p.tau[tid]._12, p.tau[tid]._13, p.tau[tid]._12, p.tau[tid]._22, p.tau[tid]._23, p.tau[tid]._13, p.tau[tid]._23, p.tau[tid]._33 };
  // mat3 L = { tau._11, tau._12, tau._13, tau._12, tau._22, tau._23, tau._13, tau._23, tau._33 };

  if (!invert(L)) {
    p.state[tid] = ROGUE;
    return;
  }
  // these are the random numbers for each direction
  /*
  vec3 vRandn = { PGD[tid].threadRNG[omp_get_thread_num()][tid].norRan(),
                  PGD[tid].threadRNG[omp_get_thread_num()][tid].norRan(),
                  PGD[tid].threadRNG[omp_get_thread_num()][tid].norRan() };
  */
  // vec3 vRandn = { 0.1f, 0.1f, 0.1f };

  // now calculate a bunch of values for the current particle
  // calculate the time derivative of the stress tensor: (tau_current - tau_old)/dt
  mat3sym tau_ddt = { (p.tau[tid]._11 - p.tau_old[tid]._11) / par_dt,
                      (p.tau[tid]._12 - p.tau_old[tid]._12) / par_dt,
                      (p.tau[tid]._13 - p.tau_old[tid]._13) / par_dt,
                      (p.tau[tid]._22 - p.tau_old[tid]._22) / par_dt,
                      (p.tau[tid]._23 - p.tau_old[tid]._23) / par_dt,
                      (p.tau[tid]._33 - p.tau_old[tid]._33) / par_dt };

  // now calculate and set the A and b matrices for an Ax = b
  // A = -I + 0.5*(-CoEps*L + dTdt*L )*par_dt;
  mat3 A = { -1.0f + 0.50f * (-CoEps * L._11 + L._11 * tau_ddt._11 + L._12 * tau_ddt._12 + L._13 * tau_ddt._13) * par_dt,
             -0.0f + 0.50f * (-CoEps * L._12 + L._12 * tau_ddt._11 + L._22 * tau_ddt._12 + L._23 * tau_ddt._13) * par_dt,
             -0.0f + 0.50f * (-CoEps * L._13 + L._13 * tau_ddt._11 + L._23 * tau_ddt._12 + L._33 * tau_ddt._13) * par_dt,
             -0.0f + 0.50f * (-CoEps * L._12 + L._11 * tau_ddt._12 + L._12 * tau_ddt._22 + L._13 * tau_ddt._23) * par_dt,
             -1.0f + 0.50f * (-CoEps * L._22 + L._12 * tau_ddt._12 + L._22 * tau_ddt._22 + L._23 * tau_ddt._23) * par_dt,
             -0.0f + 0.50f * (-CoEps * L._23 + L._13 * tau_ddt._12 + L._23 * tau_ddt._22 + L._33 * tau_ddt._23) * par_dt,
             -0.0f + 0.50f * (-CoEps * L._13 + L._11 * tau_ddt._13 + L._12 * tau_ddt._23 + L._13 * tau_ddt._33) * par_dt,
             -0.0f + 0.50f * (-CoEps * L._23 + L._12 * tau_ddt._13 + L._22 * tau_ddt._23 + L._23 * tau_ddt._33) * par_dt,
             -1.0f + 0.50f * (-CoEps * L._33 + L._13 * tau_ddt._13 + L._23 * tau_ddt._23 + L._33 * tau_ddt._33) * par_dt };

  // b = -vectFluct - 0.5*vecFluxDiv*par_dt - sqrt(CoEps*par_dt)*vecRandn;
  vec3 b = { -p.velFluct_old[tid]._1 - 0.50f * p.flux_div[tid]._1 * par_dt - std::sqrt(CoEps * par_dt) * vRandn._1,
             -p.velFluct_old[tid]._2 - 0.50f * p.flux_div[tid]._2 * par_dt - std::sqrt(CoEps * par_dt) * vRandn._2,
             -p.velFluct_old[tid]._3 - 0.50f * p.flux_div[tid]._3 * par_dt - std::sqrt(CoEps * par_dt) * vRandn._3 };

  // now prepare for the Ax=b calculation by calculating the inverted A matrix
  if (!invert(A)) {
    p.state[tid] = ROGUE;
    return;
  }

  // now do the Ax=b calculation using the inverted matrix (vecFluct = A*b)
  multiply(A, b, p.velFluct[tid]);

  // now check to see if the value is rogue or not
  if (std::abs(p.velFluct[tid]._1) >= vel_threshold || isnan(p.velFluct[tid]._1)) {
    // std::cerr << "Particle # " << p[tid].particleID << " is rogue, ";
    // std::cerr << "uFluct = " << p[tid].velFluct._1 << ", CoEps = " << p[tid].CoEps << std::endl;
    p.velFluct[tid]._1 = 0.0;
    // isActive = false;
    p.state[tid] = ROGUE;
  }
  if (std::abs(p.velFluct[tid]._2) >= vel_threshold || isnan(p.velFluct[tid]._2)) {
    // std::cerr << "Particle # " << p[tid].particleID << " is rogue, ";
    // std::cerr << "vFluct = " << p[tid].velFluct._2 << ", CoEps = " << p[tid].CoEps << std::endl;
    p.velFluct[tid]._2 = 0.0;
    // isActive = false;
    p.state[tid] = ROGUE;
  }
  if (std::abs(p.velFluct[tid]._3) >= vel_threshold || isnan(p.velFluct[tid]._3)) {
    // std::cerr << "Particle # " << p[tid].particleID << " is rogue, ";
    // std::cerr << "wFluct = " << p[tid].velFluct._3 << ", CoEps = " << p[tid].CoEps << std::endl;
    p.velFluct[tid]._3 = 0.0;
    // isActive = false;
    p.state[tid] = ROGUE;
  }

  // p.velFluct[tid]._1 = velFluct._1;
  // p.velFluct[tid]._2 = velFluct._2;
  // p.velFluct[tid]._3 = velFluct._3;

  // now update the old values to be ready for the next particle time iteration
  // the current values are already set for the next iteration by the above calculations
  // p[tid].tau_old = p[tid].tau;
  p.tau_old[tid] = p.tau[tid];
}

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


__device__ void advect(particle_array p,
                       int tid,
                       float par_dt,
                       const BC_Params &bc_param)
{
  vec3 dist{ (p.velMean[tid]._1 + p.velFluct[tid]._1) * par_dt,
             (p.velMean[tid]._2 + p.velFluct[tid]._2) * par_dt,
             (p.velMean[tid]._3 + p.velFluct[tid]._3) * par_dt };

  p.pos[tid]._1 = p.pos[tid]._1 + dist._1;
  p.pos[tid]._2 = p.pos[tid]._2 + dist._2;
  p.pos[tid]._3 = p.pos[tid]._3 + dist._3;

  boundary_conditions(p, tid, bc_param);

  p.delta_velFluct[tid]._1 = p.velFluct[tid]._1 - p.velFluct_old[tid]._1;
  p.delta_velFluct[tid]._2 = p.velFluct[tid]._2 - p.velFluct_old[tid]._2;
  p.delta_velFluct[tid]._3 = p.velFluct[tid]._3 - p.velFluct_old[tid]._3;

  p.velFluct_old[tid] = p.velFluct[tid];
}

__global__ void set_new_particle(int new_particle, particle_array p, float *d_RNG_vals)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < new_particle) {
    // makeRealizable(1.0E-6, p.tau[idx]);
    p.velFluct_old[idx]._1 = p.velFluct_old[idx]._1 * d_RNG_vals[idx];
    p.velFluct_old[idx]._2 = p.velFluct_old[idx]._2 * d_RNG_vals[idx + new_particle];
    p.velFluct_old[idx]._3 = p.velFluct_old[idx]._3 * d_RNG_vals[idx + 2 * new_particle];
  }
}

__global__ void advect_particle(particle_array d_particle_list,
                                float *d_RNG_vals,
                                const BC_Params &bc_param,
                                int length)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    if (d_particle_list.state[idx] == ACTIVE) {

      vec3 rng = { d_RNG_vals[idx], d_RNG_vals[idx + length], d_RNG_vals[idx + 2 * length] };
      solve(d_particle_list, idx, 1.0f, 0.0000001f, 10.0f, rng);

      advect(d_particle_list, idx, 1.0f, bc_param);
    }
  }
}

void Model::getNewParticle(const int &num_new_particle,
                           particle_array d_particle,
                           const QESTurbData &d_qes_turb_data,
                           const QESgrid &qes_grid,
                           RandomGenerator *random,
                           Interpolation *interpolation,
                           Partition *partition)
{
  int blockSize = 256;
  int numBlocks = (num_new_particle + blockSize - 1) / blockSize;

  particle_array d_new_particle;
  partition->allocate_device_particle_list(d_new_particle, num_new_particle);

  cudaMemset(d_new_particle.state, ACTIVE, num_new_particle * sizeof(int));
  std::vector<uint32_t> new_ID(num_new_particle);
  id_gen->get(new_ID);
  cudaMemcpy(d_new_particle.ID, new_ID.data(), num_new_particle * sizeof(uint32_t), cudaMemcpyHostToDevice);
  std::vector<vec3> new_pos(num_new_particle, { 20.0, 50.0, 70.0 });
  cudaMemcpy(d_new_particle.pos, new_pos.data(), num_new_particle * sizeof(vec3), cudaMemcpyHostToDevice);

  interpolation->get(d_new_particle, d_qes_turb_data, qes_grid, num_new_particle);

  random->create("new_particle", 3 * num_new_particle);
  random->generate("new_particle", 0.0, 1.0);

  set_new_particle<<<numBlocks, blockSize>>>(num_new_particle, d_new_particle, random->get("new_particle"));

  // cudaDeviceSynchronize();
  partition->insert(num_new_particle, d_new_particle, d_particle);

  random->destroy("new_particle");
  partition->free_device_particle_list(d_new_particle);
}


void Model::advectParticle(particle_array d_particle,
                           const int &num_particle,
                           const BC_Params &bc_param,
                           RandomGenerator *random)
{
  int blockSize = 256;
  int numBlocks = (num_particle + blockSize - 1) / blockSize;

  random->generate("advect", 0.0, 1.0);
  advect_particle<<<numBlocks, blockSize>>>(d_particle, random->get("advect"), bc_param, num_particle);
}
