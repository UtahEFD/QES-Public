#include <iostream>
#include <vector>

#include "CUDAPartitionKernel.h"

__global__ void partitionKernel(float* arr, float* lower, float* upper, int* lower_count, int* upper_count, int size, float pivot)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // When the value at the idx <= the pivot value
        // store that value in the first of the lower part array,
	// have to use atomic adds to make sure the value of the 
        // otherwise in the upper part.  When done, these will be
        // combined to form the partitioned array.
        float value = arr[idx];
        if (value <= pivot) {

            // Update the count of the last index (lower_count or
            // upper_count) with atomic add since other threads
            // are doing the same thing. This is the position in the
            // data array to store the partitioned data
            int pos = atomicAdd(lower_count, 1);
            lower[pos] = value;
        } else {
            int pos = atomicAdd(upper_count, 1);
            upper[pos] = value;
        }
    }
}

void partitionData(std::vector<float>& data, float pivot)
{
    float *d_data, *d_lower, *d_upper;
    int *d_lower_count, *d_upper_count;

    auto size = data.size();

    cudaMalloc(&d_data, size * sizeof(float));

    cudaMalloc(&d_lower, size * sizeof(float));
    cudaMalloc(&d_upper, size * sizeof(float));

    cudaMalloc(&d_lower_count, sizeof(int));
    cudaMalloc(&d_upper_count, sizeof(int));

    // Initialize counts for the lower and upper indexes to 0
    cudaMemset(d_lower_count, 0, sizeof(int));
    cudaMemset(d_upper_count, 0, sizeof(int));

    // Copy array data to device
    cudaMemcpy(d_data, data.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    partitionKernel<<<numBlocks, blockSize>>>(d_data, d_lower, d_upper, d_lower_count, d_upper_count, size, pivot);

    int h_lower_count, h_upper_count;
    cudaMemcpy(&h_lower_count, d_lower_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_upper_count, d_upper_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Copy partitioned arrays back to host
    std::vector<float> h_lower(h_lower_count);
    std::vector<float> h_upper(h_upper_count);
    cudaMemcpy(h_lower.data(), d_lower, h_lower_count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_upper.data(), d_upper, h_upper_count * sizeof(float), cudaMemcpyDeviceToHost);

    // Reconstruct the original array
    memcpy(data.data(), h_lower.data(), h_lower_count * sizeof(int));
    memcpy(data.data() + h_lower_count, h_upper.data(), h_upper_count * sizeof(int));

    cudaFree(d_data);
    cudaFree(d_lower);
    cudaFree(d_upper);
    cudaFree(d_lower_count);
    cudaFree(d_upper_count);
}
