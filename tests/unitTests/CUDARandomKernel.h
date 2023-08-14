#pragma once

#include <cuda.h>
#include <curand.h>

void genPRNOnGPU(int n, float *prngValues, float *result);

