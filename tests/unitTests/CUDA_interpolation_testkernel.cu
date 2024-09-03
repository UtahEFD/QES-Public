
#include "CUDA_interpolation_testkernel.h"

#include "util/VectorMath_CUDA.cuh"

#include "CUDA_interpolation.cuh"

__device__ __managed__ QESgrid qes_grid;

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
  cudaMalloc((void **)&d_particle_list.nuT, length * sizeof(float));
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
  // std::cout << " tau_ii  : " << p.tau._11 << ", " << p.tau._22 << ", " << p.tau._33 << std::endl;
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

  int num_cell = qes_grid.nx * qes_grid.ny * qes_grid.nz;
  std::vector<float> data;
  data.resize(num_cell, 1.0);

  if (errorCheck == cudaSuccess) {
    auto gpuStartTime = std::chrono::high_resolution_clock::now();

    // set QES grid
    qes_grid.dx = 1;
    qes_grid.dy = 1;
    qes_grid.dz = 1;

    qes_grid.nx = 400;
    qes_grid.ny = 400;
    qes_grid.nz = 400;

    int num_cell = qes_grid.nx * qes_grid.ny * qes_grid.nz;

    // Allocate particle array on the device ONLY
    particle_array d_particle_list[2];
    allocate_device_particle_list(d_particle_list[0], length);
    allocate_device_particle_list(d_particle_list[1], length);

    // initialize on the device
    cudaMemset(d_particle_list[0].state, INACTIVE, length * sizeof(int));
    cudaMemset(d_particle_list[0].ID, 0, length * sizeof(uint32_t));

    cudaMemset(d_particle_list[1].state, INACTIVE, length * sizeof(int));
    cudaMemset(d_particle_list[1].ID, 0, length * sizeof(uint32_t));

    QESWindsData d_qes_winds_data;
    copy_data_gpu(num_cell, d_qes_winds_data);

    QESTurbData d_qes_turb_data;
    copy_data_gpu(num_cell, d_qes_turb_data);

    int blockSize = 256;

    int idx = 0, alt_idx = 1;

    // call kernel
    auto kernelStartTime = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < ntest; ++k) {

      cudaMemset(d_particle_list[idx].state, ACTIVE, length * sizeof(int));

      std::vector<vec3> new_pos(length, { 20.0, 50.0, 70.0 });
      cudaMemcpy(d_particle_list[idx].pos, new_pos.data(), length * sizeof(vec3), cudaMemcpyHostToDevice);

      int num_particle = length;// h_lower_count + new_particle;
      // std::cout << num_particle << std::endl;

      int numBlocks_buffer = (length + blockSize - 1) / blockSize;
      int numBlocks_all_particle = (num_particle + blockSize - 1) / blockSize;
      int numBlocks_new_particle = (new_particle + blockSize - 1) / blockSize;

      // these indeces are used to leap-frog the lists of the particles.
      // idx = k % 2;
      // alt_idx = (k + 1) % 2;

      interpolate<<<numBlocks_all_particle, blockSize>>>(num_particle, d_particle_list[idx], d_qes_winds_data, qes_grid);
      interpolate<<<numBlocks_all_particle, blockSize>>>(num_particle, d_particle_list[idx], d_qes_turb_data, qes_grid);
      cudaDeviceSynchronize();
    }

    auto kernelEndTime = std::chrono::high_resolution_clock::now();

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

    auto gpuEndTime = std::chrono::high_resolution_clock::now();


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
