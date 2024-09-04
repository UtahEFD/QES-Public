#include "CUDA_particle_partition.cuh"

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
  cudaFree(d_particle_list.nuT);

  cudaFree(d_particle_list.pos);
  cudaFree(d_particle_list.velMean);

  cudaFree(d_particle_list.velFluct);
  cudaFree(d_particle_list.velFluct_old);
  cudaFree(d_particle_list.delta_velFluct);

  cudaFree(d_particle_list.tau);
  cudaFree(d_particle_list.tau_old);

  cudaFree(d_particle_list.flux_div);
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
    // use all fluctuation as initial condition (sigma)
    d_particle_list.velFluct_old[idx + (*lower)] = d_new_particle_list.velFluct_old[idx];
  }
}
