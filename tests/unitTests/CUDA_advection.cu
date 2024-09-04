#include "CUDA_advection.cuh"

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

__device__ void advect(particle_array p, int tid, float par_dt, const BC_Params &bc_param)
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

__global__ void advect_particle(int length, particle_array d_particle_list, float *d_RNG_vals, const BC_Params &bc_param)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    if (d_particle_list.state[idx] == ACTIVE) {

      solve(d_particle_list, idx, 1.0f, 0.0000001f, 10.0f, { d_RNG_vals[idx], d_RNG_vals[idx + length], d_RNG_vals[idx + 2 * length] });
      advect(d_particle_list, idx, 1.0f, bc_param);
    }
  }
}
