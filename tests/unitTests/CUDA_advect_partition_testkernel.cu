
#include "CUDA_advect_partition_testkernel.h"

// #include "util/VectorMath_CUDA.cuh"

#include "plume/ParticleIDGen.h"

// #include "CUDA_GLE_Solver.cuh"
//  #include "Particle.cuh"

#include "CUDA_boundary_conditions.cuh"
#include "CUDA_particle_partition.cuh"
#include "CUDA_advection.cuh"

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

__device__ __managed__ BC_Params bc_param;

void print_percentage(const float &percentage)
{
  int val = (int)(percentage * 100);
  int lpad = (int)(percentage * PBWIDTH);
  int rpad = PBWIDTH - lpad;
  printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
  fflush(stdout);
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

  ParticleIDGen *id_gen;
  id_gen = ParticleIDGen::getInstance();

  // set boundary condition
  bc_param.xStartDomain = 0;
  bc_param.yStartDomain = 0;
  bc_param.zStartDomain = 0;

  bc_param.xEndDomain = 200;
  bc_param.yEndDomain = 100;
  bc_param.zEndDomain = 140;

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
    std::cout << "buffer usage: " << std::endl;
    for (int k = 0; k < ntest; ++k) {

      cudaMemset(d_lower_count, 0, sizeof(int));
      cudaMemset(d_upper_count, 0, sizeof(int));

      cudaMemset(d_new_particle_list.state, ACTIVE, new_particle * sizeof(int));
      std::vector<uint32_t> new_ID(new_particle);
      id_gen->get(new_ID);
      cudaMemcpy(d_new_particle_list.ID, new_ID.data(), new_particle * sizeof(uint32_t), cudaMemcpyHostToDevice);
      std::vector<vec3> new_pos(new_particle, { 20.0, 50.0, 70.0 });
      cudaMemcpy(d_new_particle_list.pos, new_pos.data(), new_particle * sizeof(vec3), cudaMemcpyHostToDevice);
      std::vector<vec3> new_sig(new_particle, { 0.1, 0.1, 0.1 });
      cudaMemcpy(d_new_particle_list.velFluct_old, new_sig.data(), new_particle * sizeof(vec3), cudaMemcpyHostToDevice);


      int num_particle = length;// h_lower_count + new_particle;
      // std::cout << num_particle << std::endl;

      int numBlocks_buffer = (length + blockSize - 1) / blockSize;
      int numBlocks_all_particle = (num_particle + blockSize - 1) / blockSize;
      int numBlocks_new_particle = (new_particle + blockSize - 1) / blockSize;

      // these indeces are used to leap-frog the lists of the particles.
      idx = k % 2;
      alt_idx = (k + 1) % 2;

      curandGenerateNormal(gen, d_RNG_vals, 3 * length, 0.0, 1.0);

      partition_particle<<<numBlocks_buffer, blockSize>>>(d_particle_list[idx],
                                                          d_particle_list[alt_idx],
                                                          d_lower_count,
                                                          d_upper_count,
                                                          length);
      cudaDeviceSynchronize();

      insert_particle<<<numBlocks_new_particle, blockSize>>>(new_particle,
                                                             d_lower_count,
                                                             d_new_particle_list,
                                                             d_particle_list[idx],
                                                             length);
      cudaDeviceSynchronize();

      interpolate_particle<<<numBlocks_all_particle, blockSize>>>(num_particle, d_particle_list[idx]);
      advect_particle<<<numBlocks_all_particle, blockSize>>>(d_particle_list[idx], d_RNG_vals, bc_param, num_particle);

      // this is slower that calling devive function bc in the kernel
      // boundary_conditions<<<numBlocks_all_particle, blockSize>>>(num_particle, d_particle_list[idx]);

      cudaMemcpy(&h_lower_count, d_lower_count, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&h_upper_count, d_upper_count, sizeof(int), cudaMemcpyDeviceToHost);
      // std::cout << k << " " << h_lower_count << " " << h_upper_count << std::endl;

      cudaDeviceSynchronize();

      print_percentage((float)h_lower_count / (float)length);
    }
    std::cout << std::endl;


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
    std::cout << "buffer status: " << h_lower_count << " " << h_upper_count << " " << length << std::endl;
    print_percentage((float)h_lower_count / (float)length);
    std::cout << std::endl;
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
