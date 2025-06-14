#pragma once

#include <chrono>
#include <string>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "util/VectorMath.h"
#include "plume/ParticleIDGen.h"

#include "Particle.h"

#include "util/VectorMath_CUDA.h"
#include "plume/CUDA/Partition.h"
#include "plume/CUDA/Concentration.h"


#include <cuda.h>
#include <curand.h>

void test_gpu(const int &, const int &, const int &);
