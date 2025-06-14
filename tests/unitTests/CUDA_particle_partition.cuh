#pragma once

#include <chrono>
#include <string>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "util/VectorMath.h"
// #include "util/VectorMath_CUDA.cuh"

#include <cuda.h>
#include <curand.h>

#include "plume/Particle.h"

void allocate_device_particle_list(particle_array &d_particle_list, int length);
void free_device_particle_list(particle_array &d_particle_list);

__device__ void copy_particle(particle_array d_particle_list_left,
                              int idx_left,
                              particle_array d_particle_list_right,
                              int idx_right);

__global__ void partition_particle_reset(particle_array d_particle_list,
                                         int size);
__global__ void partition_particle_select(particle_array d_particle_list,
                                          int *lower_count,
                                          int *upper_count,
                                          int *d_sorting_index,
                                          int size);
__global__ void partition_particle_sorting(particle_array d_particle_list_left,
                                           particle_array d_particle_list_right,
                                           int *d_sorting_index,
                                           int size);

__global__ void partition_particle(particle_array d_particle_list_left,
                                   particle_array d_particle_list_right,
                                   int *lower_count,
                                   int *upper_count,
                                   int size);

__global__ void check_buffer(particle_array d_particle_list,
                             int *lower_count,
                             int *upper_count,
                             int size);

__global__ void insert_particle(int new_particle,
                                int *lower,
                                particle_array d_new_particle_list,
                                particle_array d_particle_list,
                                int length);
