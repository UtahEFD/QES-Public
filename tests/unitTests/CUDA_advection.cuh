#pragma once

#include <chrono>
#include <string>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "util/VectorMath.h"

#include <cuda.h>
#include <curand.h>

#include "Particle.h"

#include "CUDA_boundary_conditions.cuh"
#include "CUDA_particle_partition.cuh"

__global__ void advect_particle(particle_array d_particle_list,
                                float *d_RNG_vals,
                                const BC_Params &bc_param,
                                int length);
