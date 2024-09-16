#pragma once

#include <chrono>
#include <string>
#include <cstdio>
#include <algorithm>
#include <vector>

#include "util/TimerTool.h"
#include "util/VectorMath.h"
#include "util/QESFileOutput_v2.h"
#include "util/QESNetCDFOutput_v2.h"

#include "plume/IDGenerator.h"

#include <cuda.h>
#include <curand.h>

#include "CUDA_QES_Data.h"
#include "Particle.h"

void test_gpu(const int &, const int &, const int &);
