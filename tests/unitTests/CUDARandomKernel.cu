#include "CUDARandomKernel.h"

__global__ void kernelComputation(float *prngValues, float *result)
{
    int id =  (blockDim.x * blockIdx.x) + threadIdx.x;

    // Generate value between -1.0f and 1.0f
    // convert to [-1, 1] range
    result[id] = prngValues[id] * 2.0f - 1.0f;
}

void genPRNOnGPU(int n, float *prngValues, float *result)
{
    int BLOCK_SIZE = 1024;
    int GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    kernelComputation<<<GRID_SIZE, BLOCK_SIZE>>>(prngValues, result);
}
