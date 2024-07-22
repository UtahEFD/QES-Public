
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

__device__ void copy_particle(particle_array d_particle_list_left, int idx_left, particle_array d_particle_list_right, int idx_right)
{
  // some variables do not need to be copied as the copy is done at the very beginning of the timestep and
  // they will be reset by the interpolation for example

  d_particle_list_left.state[idx_left] = d_particle_list_right.state[idx_right];
  d_particle_list_left.ID[idx_left] = d_particle_list_right.ID[idx_right];

  d_particle_list_left.pos[idx_left] = d_particle_list_right.pos[idx_right];

  // d_particle_list_left.velMean[idx_left] = d_particle_list_right.velMean[idx_right];

  // d_particle_list_left.velFluct[idx_left] = d_particle_list_right.velFluct[idx_right];
  d_particle_list_left.velFluct_old[idx_left] = d_particle_list_right.velFluct_old[idx_right];
  d_particle_list_left.delta_velFluct[idx_left] = d_particle_list_right.delta_velFluct[idx_right];

  // d_particle_list_left.CoEps[idx_left] = d_particle_list_right.CoEps[idx_right];
  // d_particle_list_left.tau[idx_left] = d_particle_list_right.tau[idx_right];
  d_particle_list_left.tau_old[idx_left] = d_particle_list_right.tau_old[idx_right];
  // d_particle_list_left.flux_div[idx_left] = d_particle_list_right.flux_div[idx_right];
}

__global__ void partition_particle(particle_array d_particle_list_left, particle_array d_particle_list_right, int *lower_count, int *upper_count, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    // resert the left particle state
    d_particle_list_left.state[idx] = INACTIVE;

    // When the state at the idx active
    // copy the particle to the new array
    // have to use atomic adds to make sure the value of the
    // new index in the new array is correct
    // otherwise ignore the particle.
    int state = d_particle_list_right.state[idx];
    if (state == ACTIVE) {

      // Update the count of the last index (lower_count or
      // upper_count) with atomic add since other threads
      // are doing the same thing. This is the position in the
      // data array to store the partitioned data
      int pos = atomicAdd(lower_count, 1);
      copy_particle(d_particle_list_left, pos, d_particle_list_right, idx);
    } else {
      int pos = atomicAdd(upper_count, 1);
    }
  }
}

__global__ void check_buffer(particle_array d_particle_list, int *lower_count, int *upper_count, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    int state = d_particle_list.state[idx];
    if (state == ACTIVE) {
      int pos = atomicAdd(lower_count, 1);
    } else {
      int pos = atomicAdd(upper_count, 1);
    }
  }
}

__global__ void insert_particle(int length, int new_particle, int *lower, particle_array d_new_particle_list, particle_array d_particle_list)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < new_particle && idx + (*lower) < length) {
    d_particle_list.state[idx + (*lower)] = d_new_particle_list.state[idx];
    d_particle_list.ID[idx + (*lower)] = d_new_particle_list.ID[idx];
    d_particle_list.pos[idx + (*lower)] = d_new_particle_list.pos[idx];
    d_particle_list.velFluct_old[idx + (*lower)] = { 0.1, 0.1, 0.1 };
  }
}

__global__ void interpolate_particle(int length, particle_array d_particle_list)
{

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    if (d_particle_list.state[idx] == ACTIVE) {

      d_particle_list.velMean[idx] = { 1.0f, 0.0f, 0.0f };

      d_particle_list.CoEps[idx] = 0.1f;
      d_particle_list.tau[idx] = { 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f };
      d_particle_list.flux_div[idx] = { 0.0f, 0.0f, 0.0f };
    }
  }
}

__global__ void advect_particle(int length, particle_array d_particle_list, float *d_RNG_vals)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    // PGD[tid].interp[tid].interpValues(TGD, p[tid].pos, p[tid].tau,p[tid].flux_div, p[tid].nuT, p[tid].CoEps);
    if (d_particle_list.state[idx] == ACTIVE) {

      solve(d_particle_list, idx, 1.0f, 0.0000001f, 10.0f, { d_RNG_vals[idx], d_RNG_vals[idx + length], d_RNG_vals[idx + 2 * length] });
      advect(d_particle_list, idx, 1.0f);
    }
  }
}
__global__ void boundary_conditions(int length, particle_array d_particle_list)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    vec3 pos = d_particle_list.pos[idx];
    if (pos._1 < 0 || pos._1 > 100) {
      d_particle_list.state[idx] = INACTIVE;
    }
  }
}


void allocate_device_particle_list(particle_array &d_particle_list, int length)
{
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
}

void free_device_particle_list(particle_array &d_particle_list)
{
  cudaFree(d_particle_list.state);
  cudaFree(d_particle_list.ID);

  cudaFree(d_particle_list.CoEps);

  cudaFree(d_particle_list.pos);
  cudaFree(d_particle_list.velMean);

  cudaFree(d_particle_list.velFluct);
  cudaFree(d_particle_list.velFluct_old);
  cudaFree(d_particle_list.delta_velFluct);

  cudaFree(d_particle_list.tau);
  cudaFree(d_particle_list.tau_old);

  cudaFree(d_particle_list.flux_div);
}


void print_particle(const particle &p)
{
  std::string particle_state = "ERROR";
  switch (p.state) {
  case ACTIVE:
    particle_state = "ACTIVE";
    break;
  case INACTIVE:
    particle_state = "INACTIVE";
    break;
  case ROGUE:
    particle_state = "ROGUE";
    break;
  }
  std::cout << "--------------------------------------" << std::endl;
  std::cout << "Particle test print:" << std::endl;
  std::cout << " state   : " << particle_state << std::endl;
  std::cout << " ID      : " << p.ID << std::endl;
  std::cout << " position: " << p.pos._1 << ", " << p.pos._2 << ", " << p.pos._3 << std::endl;
  std::cout << " velocity: " << p.velMean._1 << ", " << p.velMean._2 << ", " << p.velMean._3 << std::endl;
  std::cout << " fluct   : " << p.velFluct._1 << ", " << p.velFluct._2 << ", " << p.velFluct._3 << std::endl;
}

void test_gpu(const int &ntest, const int &new_particle, const int &length)
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

  int h_lower_count = 0, h_upper_count;

  if (errorCheck == cudaSuccess) {
    auto gpuStartTime = std::chrono::high_resolution_clock::now();

    // Allocate particle array on the device ONLY
    particle_array d_particle_list[2];
    allocate_device_particle_list(d_particle_list[0], length);
    allocate_device_particle_list(d_particle_list[1], length);

    particle_array d_new_particle_list;
    allocate_device_particle_list(d_new_particle_list, new_particle);

    // initialize on the device
    cudaMemset(d_particle_list[0].state, INACTIVE, length * sizeof(int));
    cudaMemset(d_particle_list[0].ID, 0, length * sizeof(uint32_t));

    cudaMemset(d_particle_list[1].state, INACTIVE, length * sizeof(int));
    cudaMemset(d_particle_list[1].ID, 0, length * sizeof(uint32_t));

    // Allocate n floats on device to hold random numbers
    // Allocate numParticle * 3 floats on host
    int n = length * 3;
    // CUDA_CALL();
    cudaMalloc((void **)&d_RNG_vals, n * sizeof(float));

    int *d_lower_count, *d_upper_count;
    cudaMalloc(&d_lower_count, sizeof(int));
    cudaMalloc(&d_upper_count, sizeof(int));

    int blockSize = 256;

    int idx = 0, alt_idx = 1;

    // call kernel
    auto kernelStartTime = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < ntest; ++k) {

      cudaMemset(d_lower_count, 0, sizeof(int));
      cudaMemset(d_upper_count, 0, sizeof(int));

      cudaMemset(d_new_particle_list.state, ACTIVE, new_particle * sizeof(int));
      std::vector<uint32_t> new_ID(new_particle);
      id_gen->get(new_ID);
      cudaMemcpy(d_new_particle_list.ID, new_ID.data(), new_particle * sizeof(uint32_t), cudaMemcpyHostToDevice);
      std::vector<vec3> new_pos(new_particle, { 0.0, 0.0, 0.0 });
      cudaMemcpy(d_new_particle_list.pos, new_pos.data(), new_particle * sizeof(vec3), cudaMemcpyHostToDevice);

      int num_particle = length;// h_lower_count + new_particle;
      // std::cout << num_particle << std::endl;

      int numBlocks_buffer = (length + blockSize - 1) / blockSize;
      int numBlocks_all_particle = (num_particle + blockSize - 1) / blockSize;
      int numBlocks_new_particle = (new_particle + blockSize - 1) / blockSize;

      idx = k % 2;
      alt_idx = (k + 1) % 2;

      curandGenerateNormal(gen, d_RNG_vals, 3 * length, 0.0, 1.0);

      partition_particle<<<numBlocks_buffer, blockSize>>>(d_particle_list[idx], d_particle_list[alt_idx], d_lower_count, d_upper_count, length);
      cudaDeviceSynchronize();

      insert_particle<<<numBlocks_new_particle, blockSize>>>(length, new_particle, d_lower_count, d_new_particle_list, d_particle_list[idx]);
      cudaDeviceSynchronize();

      interpolate_particle<<<numBlocks_all_particle, blockSize>>>(num_particle, d_particle_list[idx]);
      advect_particle<<<numBlocks_all_particle, blockSize>>>(num_particle, d_particle_list[idx], d_RNG_vals);
      boundary_conditions<<<numBlocks_all_particle, blockSize>>>(num_particle, d_particle_list[idx]);

      cudaMemcpy(&h_lower_count, d_lower_count, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&h_upper_count, d_upper_count, sizeof(int), cudaMemcpyDeviceToHost);
      // std::cout << k << " " << h_lower_count << " " << h_upper_count << std::endl;

      cudaDeviceSynchronize();
    }

    int numBlocks_buffer = (length + blockSize - 1) / blockSize;
    cudaMemset(d_lower_count, 0, sizeof(int));
    cudaMemset(d_upper_count, 0, sizeof(int));
    check_buffer<<<numBlocks_buffer, blockSize>>>(d_particle_list[idx], d_lower_count, d_upper_count, length);

    auto kernelEndTime = std::chrono::high_resolution_clock::now();

    cudaMemcpy(&h_lower_count, d_lower_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_upper_count, d_upper_count, sizeof(int), cudaMemcpyDeviceToHost);

    // cudamemcpy back to host
    std::vector<int> particle_state(length);
    std::vector<uint32_t> particle_ID(length);
    // cudaMemcpy(isRogue.data(), d_particle_list.isRogue, length * sizeof(bool), cudaMemcpyDeviceToHost);
    // cudaMemcpy(isActive.data(), &d_particle_list.isActive, length * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(particle_state.data(), d_particle_list[idx].state, length * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(particle_ID.data(), d_particle_list[idx].ID, length * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::vector<vec3> pos(length);
    std::vector<vec3> velMean(length);
    std::vector<vec3> velFluct(length);
    // cudaMemcpy(CoEps.data(), d_particle_list.CoEps, length * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(pos.data(), d_particle_list[idx].pos, length * sizeof(vec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(velMean.data(), d_particle_list[idx].velMean, length * sizeof(vec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(velFluct.data(), d_particle_list[idx].velFluct, length * sizeof(vec3), cudaMemcpyDeviceToHost);

    std::vector<particle> particle_list(length);

    for (size_t k = 0; k < particle_list.size(); ++k) {
      particle_list[k].state = particle_state[k];
      particle_list[k].ID = particle_ID[k];
    }
    for (size_t k = 0; k < particle_list.size(); ++k) {
      particle_list[k].pos = pos[k];
    }
    for (size_t k = 0; k < particle_list.size(); ++k) {
      particle_list[k].velMean = velMean[k];
    }
    for (size_t k = 0; k < particle_list.size(); ++k) {
      particle_list[k].velFluct = velFluct[k];
    }

    // cudafree
    free_device_particle_list(d_particle_list[0]);
    free_device_particle_list(d_particle_list[1]);
    free_device_particle_list(d_new_particle_list);

    auto gpuEndTime = std::chrono::high_resolution_clock::now();


    int count = 0;
    for (auto &p : particle_list) {
      if (p.state == ACTIVE) { count++; }
    }
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "buffer status: " << h_lower_count << " " << h_upper_count << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "number of active particle: " << count << std::endl;

    print_particle(particle_list[0]);
    print_particle(particle_list[1]);

    std::cout << "--------------------------------------" << std::endl;
    std::chrono::duration<double> kernelElapsed = kernelEndTime - kernelStartTime;
    std::cout << "kernel elapsed time: " << kernelElapsed.count() << " s\n";
    std::chrono::duration<double> gpuElapsed = gpuEndTime - gpuStartTime;
    std::cout << "GPU elapsed time:    " << gpuElapsed.count() << " s\n";
  } else {
    printf("CUDA ERROR!\n");
  }
}
