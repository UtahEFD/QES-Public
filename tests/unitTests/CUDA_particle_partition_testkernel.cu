
#include "CUDA_particle_partition_testkernel.h"

#include "util/VectorMath_CUDA.cuh"

#include "plume/IDGenerator.h"

// #include "CUDA_GLE_Solver.cuh"
//  #include "Particle.cuh"

__device__ void copy_particle(particle_array d_particle_list_left, int idx_left, particle_array d_particle_list_right, int idx_right)
{
  // some variables do not need to be copied as the copy is done at the very beginning of the timestep and
  // they will be reset by the interpolation for example

  d_particle_list_left.state[idx_left] = d_particle_list_right.state[idx_right];
  d_particle_list_left.ID[idx_left] = d_particle_list_right.ID[idx_right];

  d_particle_list_left.pos[idx_left] = d_particle_list_right.pos[idx_right];

  d_particle_list_left.velMean[idx_left] = d_particle_list_right.velMean[idx_right];

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

__global__ void insert_particle(int length, int new_particle, int *lower, particle_array d_new_particle_list, particle_array d_particle_list)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < new_particle && idx + (*lower) < length) {
    d_particle_list.state[idx + (*lower)] = d_new_particle_list.state[idx];
    d_particle_list.ID[idx + (*lower)] = d_new_particle_list.ID[idx];
    d_particle_list.pos[idx + (*lower)] = { 0.0, 0.0, 0.0 };
  }
}

__global__ void set_particle(int length, particle_array d_particle_list, float *d_RNG_vals)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < length) {
    int state = d_particle_list.state[idx];
    if (state == ACTIVE) {
      d_particle_list.pos[idx]._1++;
      float value = d_RNG_vals[idx];
      if (value > 0.8) {
        d_particle_list.state[idx] = INACTIVE;
      }
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
  std::cout << "--------------------------------------" << std::endl;
  std::cout << "Particle test print:" << std::endl;
  std::cout << " state   : " << p.isActive << ", " << p.isRogue << std::endl;
  std::cout << " ID      : " << p.ID << std::endl;
  std::cout << " position: " << p.pos._1 << ", " << p.pos._2 << ", " << p.pos._3 << std::endl;
  std::cout << " velocity: " << p.velMean._1 << ", " << p.velMean._2 << ", " << p.velMean._3 << std::endl;
  std::cout << " fluct   : " << p.velFluct._1 << ", " << p.velFluct._2 << ", " << p.velFluct._3 << std::endl;
  std::cout << "--------------------------------------" << std::endl;
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


  int *d_lower_count, *d_upper_count;

  if (errorCheck == cudaSuccess) {
    // temp
    // print_particle(particle_list[2]);
    auto gpuStartTime = std::chrono::high_resolution_clock::now();

    // Allocate particle array on the device ONLY
    particle_array d_particle_list_odd;
    allocate_device_particle_list(d_particle_list_odd, length);
    particle_array d_particle_list_even;
    allocate_device_particle_list(d_particle_list_even, length);
    particle_array d_new_particle_list;
    allocate_device_particle_list(d_new_particle_list, new_particle);
    // initialize on the device
    cudaMemset(d_particle_list_odd.state, INACTIVE, length * sizeof(int));
    cudaMemset(d_particle_list_odd.ID, 0, length * sizeof(uint32_t));

    cudaMemset(d_particle_list_odd.state, INACTIVE, length * sizeof(int));
    cudaMemset(d_particle_list_odd.ID, 0, length * sizeof(uint32_t));

    // Allocate n floats on device to hold random numbers
    // Allocate numParticle * 3 floats on host
    // CUDA_CALL();
    cudaMalloc((void **)&d_RNG_vals, length * sizeof(float));

    cudaMalloc(&d_lower_count, sizeof(int));
    cudaMalloc(&d_upper_count, sizeof(int));

    // call kernel
    auto kernelStartTime = std::chrono::high_resolution_clock::now();
    int h_lower_count = 0, h_upper_count;

    int blockSize = 256;
    for (int k = 0; k < ntest; ++k) {

      cudaMemset(d_lower_count, 0, sizeof(int));
      cudaMemset(d_upper_count, 0, sizeof(int));

      cudaMemset(d_new_particle_list.state, ACTIVE, new_particle * sizeof(int));
      std::vector<uint32_t> new_ID(new_particle);
      id_gen->get(new_ID);
      cudaMemcpy(d_new_particle_list.ID, new_ID.data(), new_particle * sizeof(uint32_t), cudaMemcpyHostToDevice);

      int num_particle = length;// h_lower_count + new_particle;
      // std::cout << num_particle << std::endl;

      int numBlocks_buffer = (length + blockSize - 1) / blockSize;
      int numBlocks_all_particle = (num_particle + blockSize - 1) / blockSize;
      int numBlocks_new_particle = (new_particle + blockSize - 1) / blockSize;


      curandGenerateUniform(gen, d_RNG_vals, num_particle);
      if (k % 2 == 0) {
        partition_particle<<<numBlocks_buffer, blockSize>>>(d_particle_list_even, d_particle_list_odd, d_lower_count, d_upper_count, length);
        cudaDeviceSynchronize();

        insert_particle<<<numBlocks_new_particle, blockSize>>>(length, new_particle, d_lower_count, d_new_particle_list, d_particle_list_even);
        cudaDeviceSynchronize();

        set_particle<<<numBlocks_all_particle, blockSize>>>(num_particle, d_particle_list_even, d_RNG_vals);
        // set_particle<<<numBlocks1, blockSize>>>(length, d_particle_list_even, d_RNG_vals);
      } else {
        partition_particle<<<numBlocks_buffer, blockSize>>>(d_particle_list_odd, d_particle_list_even, d_lower_count, d_upper_count, length);
        cudaDeviceSynchronize();

        insert_particle<<<numBlocks_new_particle, blockSize>>>(length, new_particle, d_lower_count, d_new_particle_list, d_particle_list_odd);
        cudaDeviceSynchronize();

        set_particle<<<numBlocks_all_particle, blockSize>>>(num_particle, d_particle_list_odd, d_RNG_vals);
        // set_particle<<<numBlocks1, blockSize>>>(length, d_particle_list_odd, d_RNG_vals);
      }
      cudaMemcpy(&h_lower_count, d_lower_count, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&h_upper_count, d_upper_count, sizeof(int), cudaMemcpyDeviceToHost);
      // std::cout << k << " " << h_lower_count << " " << h_upper_count << std::endl;

      cudaDeviceSynchronize();
    }
    cudaMemcpy(&h_lower_count, d_lower_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_upper_count, d_upper_count, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << ntest << " " << h_lower_count << " " << h_upper_count << std::endl;

    auto kernelEndTime = std::chrono::high_resolution_clock::now();

    // copy relevant quantity back to host
    std::vector<int> particle_state(length);
    std::vector<uint32_t> particle_ID(length);
    std::vector<vec3> pos(length);

    // cudamemcpy back to host
    // cudaMemcpy(isRogue.data(), d_particle_list.isRogue, length * sizeof(bool), cudaMemcpyDeviceToHost);
    // cudaMemcpy(isActive.data(), &d_particle_list.isActive, length * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(particle_state.data(), d_particle_list_odd.state, length * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(particle_ID.data(), d_particle_list_odd.ID, length * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // cudaMemcpy(CoEps.data(), d_particle_list.CoEps, length * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(pos.data(), d_particle_list_odd.pos, length * sizeof(vec3), cudaMemcpyDeviceToHost);
    // cudaMemcpy(velMean.data(), d_particle_list.velMean, length * sizeof(vec3), cudaMemcpyDeviceToHost);
    // cudaMemcpy(velFluct.data(), d_particle_list.velFluct, length * sizeof(vec3), cudaMemcpyDeviceToHost);

    std::vector<particle> particle_list(length);

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

    for (size_t k = 0; k < particle_list.size(); ++k) {
      particle_list[k].pos = pos[k];
    }


    // cudafree
    free_device_particle_list(d_particle_list_odd);
    free_device_particle_list(d_particle_list_even);
    free_device_particle_list(d_new_particle_list);

    auto gpuEndTime = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> kernelElapsed = kernelEndTime - kernelStartTime;
    std::cout << "kernel  elapsed time: " << kernelElapsed.count() << " s\n";
    std::chrono::duration<double> gpuElapsed = gpuEndTime - gpuStartTime;
    std::cout << "GPU  elapsed time: " << gpuElapsed.count() << " s\n";

    print_particle(particle_list[0]);

  } else {
    printf("CUDA ERROR!\n");
  }
}
