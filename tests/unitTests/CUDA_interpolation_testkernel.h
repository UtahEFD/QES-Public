#pragma once

#include <chrono>
#include <string>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "util/VectorMath.h"
#include "Particle.h"

#include <cuda.h>
#include <curand.h>

#include "plume/cuda/QES_data.h"


void test_gpu(const int &, const int &, const int &);
