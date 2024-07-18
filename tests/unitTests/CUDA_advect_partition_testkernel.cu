
#include "CUDA_advect_partition_testkernel.h"

#include "util/VectorMath_CUDA.cuh"

#include "plume/IDGenerator.h"

// #include "CUDA_GLE_Solver.cuh"
//  #include "Particle.cuh"

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

__device__ void advect(particle_array p, int tid, float par_dt)
{
  vec3 dist{ (p.velMean[tid]._1 + p.velFluct[tid]._1) * par_dt,
             (p.velMean[tid]._2 + p.velFluct[tid]._2) * par_dt,
             (p.velMean[tid]._3 + p.velFluct[tid]._3) * par_dt };

  p.pos[tid]._1 = p.pos[tid]._1 + dist._1;
  p.pos[tid]._2 = p.pos[tid]._2 + dist._2;
  p.pos[tid]._3 = p.pos[tid]._3 + dist._3;

  p.delta_velFluct[tid]._1 = p.velFluct[tid]._1 - p.velFluct_old[tid]._1;
  p.delta_velFluct[tid]._2 = p.velFluct[tid]._2 - p.velFluct_old[tid]._2;
  p.delta_velFluct[tid]._3 = p.velFluct[tid]._3 - p.velFluct_old[tid]._3;

  p.velFluct_old[tid] = p.velFluct[tid];
}

__global__ void partitionKernel(float *arr, float *lower, float *upper, int *lower_count, int *upper_count, int size, float pivot)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    // When the value at the idx <= the pivot value
    // store that value in the first of the lower part array,
    // have to use atomic adds to make sure the value of the
    // otherwise in the upper part.  When done, these will be
    // combined to form the partitioned array.
    float value = arr[idx];
    if (value <= pivot) {

      // Update the count of the last index (lower_count or
      // upper_count) with atomic add since other threads
      // are doing the same thing. This is the position in the
      // data array to store the partitioned data
      int pos = atomicAdd(lower_count, 1);
      lower[pos] = value;
    } else {
      int pos = atomicAdd(upper_count, 1);
      upper[pos] = value;
    }
  }
}

__global__ void testCUDA_advection(int length, particle_array d_particle_list, float *d_RNG_vals)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int tid = index; tid < length; tid += stride) {
    // PGD[tid].interp[tid].interpValues(TGD, p[tid].pos, p[tid].tau,p[tid].flux_div, p[tid].nuT, p[tid].CoEps);
    if (d_particle_list.state[tid] == ACTIVE) {

      d_particle_list.tau[tid] = { 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f };
      d_particle_list.flux_div[tid] = { 0.0f, 0.0f, 0.0f };


      solve(d_particle_list, tid, 1.0f, 0.0000001f, 10.0f, { d_RNG_vals[tid], d_RNG_vals[tid + length], d_RNG_vals[tid + 2 * length] });
      advect(d_particle_list, tid, 1.0f);
    }
  }
  return;
}

void print_particle(const particle &p)
{
  std::cout << "--------------------------------------" << std::endl;
  std::cout << "Particle test print:" << std::endl;
  std::cout << " state   : " << p.isActive << ", " << p.isRogue << std::endl;
  std::cout << " ID      : " << p.ID << std::endl;
  std::cout << " position: " << p.pos._1 << ", " << p.pos._2 << ", " << p.pos._3 << std::endl;
  std::cout << " velocity: " << p.velMean._1 << ", " << p.velMean._2 << ", " << p.velMean._3 << std::endl;
  std::cout << " fluct   : " << p.velFluct._1 << ", " << p.velFluct._2 << ", " << p.velFluct._3 << std::endl;
  std::cout << "--------------------------------------" << std::endl;
}

void test_gpu(const int &length)
{
  int gpuID = 0;
  cudaError_t errorCheck = cudaGetDevice(&gpuID);

  int blockCount = 1;
  cudaDeviceGetAttribute(&blockCount, cudaDevAttrMultiProcessorCount, gpuID);
  // std::cout << blockCount << std::endl;

  int threadsPerBlock = 128;
  cudaDeviceGetAttribute(&threadsPerBlock, cudaDevAttrMaxThreadsPerBlock, gpuID);
  // std::cout << threadsPerBlock << std::endl;

  curandGenerator_t gen;
  float *d_RNG_vals;

  // Create pseudo-random number generator
  // CURAND_CALL(
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

  // Set the seed --- not sure how we'll do this yet in general
  // CURAND_CALL(
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

  IDGenerator *id_gen;
  id_gen = IDGenerator::getInstance();

  int blockSize = threadsPerBlock;
  dim3 numberOfThreadsPerBlock(blockSize, 1, 1);
  dim3 numberOfBlocks(ceil(length / (float)(blockSize)), 1, 1);

  particle tmp;

  tmp.isRogue = false;
  tmp.isActive = true;
  tmp.CoEps = 0.1f;
  tmp.pos = { 0.0f, 0.0f, 0.0f };
  tmp.velMean = { 1.0f, 2.0f, -1.0f };
  tmp.velFluct = { 0.0f, 0.0f, 0.0f };
  tmp.tau = { 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f };
  tmp.fluxDiv = { 0.0f, 0.0f, 0.0f };


  std::vector<particle> particle_list;
  particle_list.resize(length, tmp);

  for (auto &p : particle_list) {
    p.ID = id_gen->get();
  }

  /*std::vector<bool> isRogue(particle_list.size());
  for (size_t k = 0; k < particle_list.size(); ++k) {
    isRogue[k] = particle_list[k].isRogue;
  }
  std::vector<bool> isActive(particle_list.size());
  for (size_t k = 0; k < particle_list.size(); ++k) {
    isActive[k] = particle_list[k].isActive;
  }*/
  std::vector<int> particle_state(particle_list.size());
  for (size_t k = 0; k < particle_list.size(); ++k) {
    if (particle_list[k].isActive)
      particle_state[k] = ACTIVE;
    else if (particle_list[k].isActive && !particle_list[k].isRogue)
      particle_state[k] = INACTIVE;
    else
      particle_state[k] = ROGUE;
  }

  std::vector<uint32_t> particle_ID(particle_list.size());
  for (size_t k = 0; k < particle_list.size(); ++k) {
    particle_ID[k] = particle_list[k].ID;
  }

  std::vector<float> CoEps(particle_list.size());
  for (size_t k = 0; k < particle_list.size(); ++k) {
    CoEps[k] = particle_list[k].CoEps;
  }

  std::vector<vec3> pos(particle_list.size());
  for (size_t k = 0; k < particle_list.size(); ++k) {
    pos[k] = particle_list[k].pos;
  }
  std::vector<vec3> velMean(particle_list.size());
  for (size_t k = 0; k < particle_list.size(); ++k) {
    velMean[k] = particle_list[k].velMean;
  }
  std::vector<vec3> velFluct(particle_list.size());
  for (size_t k = 0; k < particle_list.size(); ++k) {
    velFluct[k] = particle_list[k].velFluct;
  }
  std::vector<mat3sym> tau(particle_list.size());
  for (size_t k = 0; k < particle_list.size(); ++k) {
    tau[k] = particle_list[k].tau;
  }

  if (errorCheck == cudaSuccess) {
    // temp
    print_particle(particle_list[2]);

    particle_array d_particle_list;

    auto gpuStartTime = std::chrono::high_resolution_clock::now();

    // cudaMalloc((void **)&d_particle_list.isRogue, length * sizeof(bool));
    // cudaMalloc((void **)&d_particle_list.isActive, length * sizeof(bool));

    // Allocate n floats on device to hold random numbers
    // Allocate numParticle * 3 floats on host
    int n = length * 3;
    // CUDA_CALL();
    cudaMalloc((void **)&d_RNG_vals, n * sizeof(float));

    // Allocate particle list on the GPU
    cudaMalloc((void **)&d_particle_list.state, length * sizeof(int));
    cudaMalloc((void **)&d_particle_list.ID, length * sizeof(uint32_t));

    cudaMalloc((void **)&d_particle_list.pos, length * sizeof(vec3));
    cudaMalloc((void **)&d_particle_list.velMean, length * sizeof(vec3));

    cudaMalloc((void **)&d_particle_list.velFluct, length * sizeof(vec3));
    cudaMalloc((void **)&d_particle_list.velFluct_old, length * sizeof(vec3));
    cudaMalloc((void **)&d_particle_list.delta_velFluct, length * sizeof(vec3));

    cudaMalloc((void **)&d_particle_list.CoEps, length * sizeof(float));
    cudaMalloc((void **)&d_particle_list.tau, length * sizeof(mat3sym));
    cudaMalloc((void **)&d_particle_list.tau_old, length * sizeof(mat3sym));

    cudaMalloc((void **)&d_particle_list.flux_div, length * sizeof(vec3));

    // copy to the device
    // cudaMemcpy(d_particle_list.isRogue, isRogue.data(), length * sizeof(bool), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_particle_list.isActive, isActive.data(), length * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particle_list.state, particle_state.data(), length * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particle_list.ID, particle_ID.data(), length * sizeof(uint32_t), cudaMemcpyHostToDevice);

    cudaMemcpy(d_particle_list.CoEps, CoEps.data(), length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particle_list.tau, tau.data(), length * sizeof(mat3sym), cudaMemcpyHostToDevice);

    cudaMemcpy(d_particle_list.pos, pos.data(), length * sizeof(vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particle_list.velMean, velMean.data(), length * sizeof(vec3), cudaMemcpyHostToDevice);


    // call kernel
    auto kernelStartTime = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < 1E4; ++k) {
      curandGenerateNormal(gen, d_RNG_vals, n, 0.0, 1.0);
      testCUDA_advection<<<numberOfBlocks, numberOfThreadsPerBlock>>>(length, d_particle_list, d_RNG_vals);
      cudaDeviceSynchronize();
    }
    auto kernelEndTime = std::chrono::high_resolution_clock::now();

    // cudamemcpy back to host
    // cudaMemcpy(isRogue.data(), d_particle_list.isRogue, length * sizeof(bool), cudaMemcpyDeviceToHost);
    // cudaMemcpy(isActive.data(), &d_particle_list.isActive, length * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(particle_state.data(), d_particle_list.state, length * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(particle_ID.data(), d_particle_list.ID, length * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // cudaMemcpy(CoEps.data(), d_particle_list.CoEps, length * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(pos.data(), d_particle_list.pos, length * sizeof(vec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(velMean.data(), d_particle_list.velMean, length * sizeof(vec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(velFluct.data(), d_particle_list.velFluct, length * sizeof(vec3), cudaMemcpyDeviceToHost);


    for (size_t k = 0; k < particle_list.size(); ++k) {
      particle_list[k].pos = pos[k];
    }
    for (size_t k = 0; k < particle_list.size(); ++k) {
      particle_list[k].velMean = velMean[k];
    }
    for (size_t k = 0; k < particle_list.size(); ++k) {
      particle_list[k].velFluct = velFluct[k];
    }

    for (size_t k = 0; k < particle_list.size(); ++k) {
      particle_list[k].ID = particle_ID[k];
      if (particle_state[k] == ACTIVE) {
        particle_list[k].isActive = true;
        particle_list[k].isRogue = false;
      } else if (particle_state[k] == INACTIVE) {
        particle_list[k].isActive = false;
        particle_list[k].isRogue = false;
      } else if (particle_state[k] == ROGUE) {
        particle_list[k].isRogue = true;
        particle_list[k].isActive = false;
      } else {
        // unknown state -> should not happend
      }
    }

    // cudafree
    cudaFree(d_particle_list.state);

    cudaFree(d_particle_list.CoEps);

    cudaFree(d_particle_list.pos);
    cudaFree(d_particle_list.velMean);

    cudaFree(d_particle_list.velFluct);
    cudaFree(d_particle_list.velFluct_old);
    cudaFree(d_particle_list.delta_velFluct);

    cudaFree(d_particle_list.tau);
    cudaFree(d_particle_list.tau_old);

    cudaFree(d_particle_list.flux_div);

    auto gpuEndTime = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> kernelElapsed = kernelEndTime - kernelStartTime;
    std::cout << "kernel  elapsed time: " << kernelElapsed.count() << " s\n";
    std::chrono::duration<double> gpuElapsed = gpuEndTime - gpuStartTime;
    std::cout << "GPU  elapsed time: " << gpuElapsed.count() << " s\n";

    print_particle(particle_list[0]);
    print_particle(particle_list[1]);

  } else {
    printf("CUDA ERROR!\n");
  }
}
