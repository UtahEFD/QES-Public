
#include "CUDA_particle_partition_testkernel.h"

#include "util/VectorMath_CUDA.cuh"

#include "plume/IDGenerator.h"

// #include "CUDA_GLE_Solver.cuh"
//  #include "Particle.cuh"

__global__ void partition_particle(particle_array d_particle_list_left, particle_array d_particle_list_right, int *lower_count, int *upper_count, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    // When the value at the idx <= the pivot value
    // store that value in the first of the lower part array,
    // have to use atomic adds to make sure the value of the
    // otherwise in the upper part.  When done, these will be
    // combined to form the partitioned array.
    int state = d_particle_list_right.state[idx];
    if (state == ACTIVE) {

      // Update the count of the last index (lower_count or
      // upper_count) with atomic add since other threads
      // are doing the same thing. This is the position in the
      // data array to store the partitioned data
      int pos = atomicAdd(lower_count, 1);
      d_particle_list_left.state[pos] = d_particle_list_right.state[idx];
      d_particle_list_left.ID[pos] = d_particle_list_right.ID[idx];
      d_particle_list_left.pos[pos] = d_particle_list_right.pos[idx];

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

void test_gpu(const int &ntest, const int &new_particle)
{

  const int length = 100000;

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

  particle tmp;

  tmp.isRogue = false;
  tmp.isActive = false;
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

  int *d_lower_count, *d_upper_count;

  if (errorCheck == cudaSuccess) {
    // temp
    print_particle(particle_list[2]);
    auto gpuStartTime = std::chrono::high_resolution_clock::now();

    particle_array d_particle_list_odd;
    allocate_device_particle_list(d_particle_list_odd, length);
    particle_array d_particle_list_even;
    allocate_device_particle_list(d_particle_list_even, length);
    particle_array d_new_particle_list;
    allocate_device_particle_list(d_new_particle_list, new_particle);


    // cudaMalloc((void **)&d_particle_list.isRogue, length * sizeof(bool));
    // cudaMalloc((void **)&d_particle_list.isActive, length * sizeof(bool));

    // Allocate n floats on device to hold random numbers
    // Allocate numParticle * 3 floats on host
    // CUDA_CALL();
    cudaMalloc((void **)&d_RNG_vals, length * sizeof(float));

    cudaMalloc(&d_lower_count, sizeof(int));
    cudaMalloc(&d_upper_count, sizeof(int));

    // copy to the device
    // cudaMemcpy(d_particle_list.isRogue, isRogue.data(), length * sizeof(bool), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_particle_list.isActive, isActive.data(), length * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemset(d_particle_list_even.state, INACTIVE, length * sizeof(int));
    cudaMemset(d_particle_list_even.ID, 0, length * sizeof(uint32_t));

    cudaMemset(d_particle_list_odd.state, INACTIVE, length * sizeof(int));
    cudaMemset(d_particle_list_even.ID, 0, length * sizeof(uint32_t));

    // cudaMemcpy(d_particle_list.CoEps, CoEps.data(), length * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_particle_list.tau, tau.data(), length * sizeof(mat3sym), cudaMemcpyHostToDevice);

    // cudaMemcpy(d_particle_list.pos, pos.data(), length * sizeof(vec3), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_particle_list.velMean, velMean.data(), length * sizeof(vec3), cudaMemcpyHostToDevice);


    // call kernel
    auto kernelStartTime = std::chrono::high_resolution_clock::now();
    int h_lower_count, h_upper_count;

    int blockSize = 256;
    int numBlocks = (length + blockSize - 1) / blockSize;
    int numBlocks2 = (new_particle + blockSize - 1) / blockSize;
    for (int k = 0; k < ntest; ++k) {

      cudaMemset(d_new_particle_list.state, ACTIVE, new_particle * sizeof(int));
      std::vector<uint32_t> new_ID(new_particle);
      id_gen->get(new_ID);
      cudaMemcpy(d_new_particle_list.ID, particle_ID.data(), new_particle * sizeof(uint32_t), cudaMemcpyHostToDevice);

      curandGenerateUniform(gen, d_RNG_vals, length);
      if (k == 0) {
        insert_particle<<<numBlocks2, blockSize>>>(length, new_particle, d_lower_count, d_new_particle_list, d_particle_list_even);
        set_particle<<<numBlocks, blockSize>>>(length, d_particle_list_even, d_RNG_vals);
      } else if (k % 2 == 0) {
        partition_particle<<<numBlocks, blockSize>>>(d_particle_list_even, d_particle_list_odd, d_lower_count, d_upper_count, length);
        cudaDeviceSynchronize();
        insert_particle<<<numBlocks2, blockSize>>>(length, new_particle, d_lower_count, d_new_particle_list, d_particle_list_even);
        set_particle<<<numBlocks, blockSize>>>(length, d_particle_list_even, d_RNG_vals);
      } else {
        partition_particle<<<numBlocks, blockSize>>>(d_particle_list_odd, d_particle_list_even, d_lower_count, d_upper_count, length);
        cudaDeviceSynchronize();
        insert_particle<<<numBlocks2, blockSize>>>(length, new_particle, d_lower_count, d_new_particle_list, d_particle_list_odd);
        set_particle<<<numBlocks, blockSize>>>(length, d_particle_list_odd, d_RNG_vals);
      }
      cudaDeviceSynchronize();
      cudaMemcpy(&h_lower_count, d_lower_count, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&h_upper_count, d_upper_count, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemset(d_lower_count, 0, sizeof(int));
      cudaMemset(d_upper_count, 0, sizeof(int));
      if (k % 10 == 0)
        std::cout << k << " " << h_lower_count << " " << h_upper_count << std::endl;
    }
    std::cout << ntest << " " << h_lower_count << " " << h_upper_count << std::endl;

    auto kernelEndTime = std::chrono::high_resolution_clock::now();


    // cudamemcpy back to host
    // cudaMemcpy(isRogue.data(), d_particle_list.isRogue, length * sizeof(bool), cudaMemcpyDeviceToHost);
    // cudaMemcpy(isActive.data(), &d_particle_list.isActive, length * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(particle_state.data(), d_particle_list_odd.state, length * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(particle_ID.data(), d_particle_list_odd.ID, length * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // cudaMemcpy(CoEps.data(), d_particle_list.CoEps, length * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(pos.data(), d_particle_list_odd.pos, length * sizeof(vec3), cudaMemcpyDeviceToHost);
    // cudaMemcpy(velMean.data(), d_particle_list.velMean, length * sizeof(vec3), cudaMemcpyDeviceToHost);
    // cudaMemcpy(velFluct.data(), d_particle_list.velFluct, length * sizeof(vec3), cudaMemcpyDeviceToHost);


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
