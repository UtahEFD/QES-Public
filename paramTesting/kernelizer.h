#pragma once

#include "DataA.h"
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <chrono>

#define X 10000
#define Y 10000
#define DIMX 5.0f
#define DIMY 5.0f

using std::vector;

void doTheGPU(vector<DataA> datAs);

#define cudaCheck(x) _cudaCheck(x, #x ,__FILE__, __LINE__)

template<typename T>
void _cudaCheck(T e, const char* func, const char* call, const int line);
