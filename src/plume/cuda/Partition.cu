/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Behnam Bozorgmehr
 * Copyright (c) 2024 Jeremy A. Gibbs
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Eric R. Pardyjak
 * Copyright (c) 2024 Zachary Patterson
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Lucas Ulmer
 * Copyright (c) 2024 Pete Willemsen
 *
 * This file is part of QES-Plume
 *
 * GPL-3.0 License
 *
 * QES-Plume is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Plume is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Plume. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/** @file
 * @brief
 */

#include "Partition.h"

__device__ void copy_particle(particle_array d_particle_list_left,
                              int idx_left,
                              particle_array d_particle_list_right,
                              int idx_right)
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

__global__ void partition_particle_select(particle_array d_particle_list,
                                          int *lower_count,
                                          int *upper_count,
                                          int *d_sorting_index,
                                          int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    // When the state at the idx active
    // copy the particle to the new array
    // have to use atomic adds to make sure the value of the
    // new index in the new array is correct
    // otherwise ignore the particle.
    int state = d_particle_list.state[idx];
    if (state == ACTIVE) {

      // Update the count of the last index (lower_count or
      // upper_count) with atomic add since other threads
      // are doing the same thing. This is the position in the
      // data array to store the partitioned data
      int pos = atomicAdd(lower_count, 1);
      d_sorting_index[idx] = pos;
    } else {
      int pos = atomicAdd(upper_count, 1);
      // d_sorting_index[idx] = -1;
    }
  }
}

__global__ void partition_particle_reset(particle_array d_particle_list,
                                         int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    // resert the left particle state
    d_particle_list.state[idx] = INACTIVE;
  }
}

__global__ void partition_particle_sorting(particle_array d_particle_list_left,
                                           particle_array d_particle_list_right,
                                           int *d_sorting_index,
                                           int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    if (d_sorting_index[idx] >= 0) {
      copy_particle(d_particle_list_left, d_sorting_index[idx], d_particle_list_right, idx);
    }
  }
}

__global__ void partition_particle(particle_array d_particle_list_left,
                                   particle_array d_particle_list_right,
                                   int *lower_count,
                                   int *upper_count,
                                   int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    // resert the left particle state
    // d_particle_list_left.state[idx] = INACTIVE;

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
    d_particle_list_right.state[idx] = INACTIVE;
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

__global__ void insert_particle(int new_particle,
                                int *lower,
                                particle_array d_new_particle_list,
                                particle_array d_particle_list,
                                int length)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < new_particle && idx + (*lower) < length) {
    d_particle_list.state[idx + (*lower)] = d_new_particle_list.state[idx];
    d_particle_list.ID[idx + (*lower)] = d_new_particle_list.ID[idx];
    d_particle_list.pos[idx + (*lower)] = d_new_particle_list.pos[idx];
    //  use all fluctuation as initial condition (sigma)
    d_particle_list.velFluct_old[idx + (*lower)] = d_new_particle_list.velFluct_old[idx];
    d_particle_list.tau[idx + (*lower)] = d_new_particle_list.tau[idx];
  }
}

void Partition::allocate_device()
{
  cudaError_t errorCheck = cudaGetDevice(&m_gpuID);
  if (errorCheck == cudaSuccess) {
    // std::cout << "allocate partition working device variables" << std::endl;
    cudaMalloc(&d_lower_count, sizeof(int));
    cudaMalloc(&d_upper_count, sizeof(int));
    cudaMalloc((void **)&d_sorting_index, m_length * sizeof(int));
  } else {
    std::cerr << "CUDA ERROR!" << std::endl;
    exit(1);
  }
}
void Partition::free_device()
{
  // std::cout << "free partition working device variables" << std::endl;
  cudaFree(d_lower_count);
  cudaFree(d_upper_count);
  cudaFree(d_sorting_index);
}

void Partition::allocate_device_particle_list(particle_array &d_particle_list, const int &length)
{
  d_particle_list.length = length;

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

  cudaMemset(d_particle_list.state, INACTIVE, length * sizeof(int));
  cudaMemset(d_particle_list.ID, 0, length * sizeof(uint32_t));
}

void Partition::free_device_particle_list(particle_array &d_particle_list)
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

int Partition::run(int k, particle_array d_particle[])
{
  if (d_particle[0].length != m_length || d_particle[1].length != m_length) {
    std::cerr << "ERROR PARTICLE LIST WRONG SIZE" << std::endl;
    exit(1);
  }

  int blockSize = 256;
  int numBlocks = (m_length + blockSize - 1) / blockSize;

  // these indeces are used to leap-frog the lists of the particles.
  int idx = k % 2;
  int alt_idx = (k + 1) % 2;

  cudaMemset(d_lower_count, 0, sizeof(int));
  cudaMemset(d_upper_count, 0, sizeof(int));

  cudaMemset(d_sorting_index, -1, m_length * sizeof(int));

  partition_particle_select<<<numBlocks, blockSize>>>(d_particle[alt_idx],
                                                      d_lower_count,
                                                      d_upper_count,
                                                      d_sorting_index,
                                                      m_length);
  partition_particle_reset<<<numBlocks, blockSize>>>(d_particle[idx],
                                                     m_length);
  partition_particle_sorting<<<numBlocks, blockSize>>>(d_particle[idx],
                                                       d_particle[alt_idx],
                                                       d_sorting_index,
                                                       m_length);

  cudaMemcpy(&h_lower_count, d_lower_count, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_upper_count, d_upper_count, sizeof(int), cudaMemcpyDeviceToHost);

  return idx;
}


void Partition::insert(int new_particle, particle_array d_new_particle, particle_array d_particle)
{
  if (d_particle.length != m_length) {
    std::cerr << "ERROR PARTICLE LIST WRONG SIZE" << std::endl;
    exit(1);
  }

  int blockSize = 256;
  int numBlocks = (new_particle + blockSize - 1) / blockSize;

  insert_particle<<<numBlocks, blockSize>>>(new_particle,
                                            d_lower_count,
                                            d_new_particle,
                                            d_particle,
                                            m_length);
}

void Partition::check(particle_array d_particle, int &h_active_count, int &h_empty_count)
{
  if (d_particle.length != m_length) {
    std::cerr << "ERROR PARTICLE LIST WRONG SIZE" << std::endl;
    exit(1);
  }

  int blockSize = 256;
  int numBlocks = (m_length + blockSize - 1) / blockSize;

  int *d_active_count, *d_empty_count;
  cudaMalloc(&d_active_count, sizeof(int));
  cudaMalloc(&d_empty_count, sizeof(int));

  check_buffer<<<numBlocks, blockSize>>>(d_particle, d_active_count, d_empty_count, m_length);

  cudaMemcpy(&h_active_count, d_active_count, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_empty_count, d_empty_count, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_active_count);
  cudaFree(d_active_count);
}
