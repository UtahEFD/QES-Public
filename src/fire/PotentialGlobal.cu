#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <cmath>
#include <iostream>
#include "Fire.h"

// CUDA error check macro
#define CUDA_CHECK(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__;     \
            std::cerr << " " << cudaGetErrorString(err) << std::endl;         \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

// Kernel to reset the potential fields
__global__ void reset_potential(float* Pot_u, float* Pot_v, float* Pot_w, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        Pot_u[idx] = 0.0f;
        Pot_v[idx] = 0.0f;
        Pot_w[idx] = 0.0f;
    }
}

// Kernel to compute heat release
__global__ void compute_heat_release(
    int nx, int ny, 
    const int* burn_flag, 
    const float* H0_array, 
    float dx, float dy,
    float* H0_sum, int* icent, int* jcent, int* counter) 
{
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockIdx.y * blockDim.y + threadIdx.y;

    if (ii < nx - 1 && jj < ny - 1) {
        int id = ii + jj * (nx - 1);
        if (burn_flag[id] == 1) {
            atomicAdd(H0_sum, H0_array[id]);
            atomicAdd(icent, ii);
            atomicAdd(jcent, jj);
            atomicAdd(counter, 1);
        }
    }
}

// Kernel for calculating potential fields
__global__ void calculate_potential(
    float *Pot_u, float *Pot_v, float *Pot_w, 
    const float *z_mix, const float *H0, 
    const float *u_r, const float *u_z, 
    int nx, int ny, int nz, float dx, float dy, float dz, 
    float drStar, float dzStar, float L_c, float U_c
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx-1 && j < ny-1 && k < nz-2) {
        int cell_cent = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);

        // Initialize variables
        float deltaX = i * dx / L_c;
        float deltaY = j * dy / L_c;
        float z_k = (k * dz) / L_c;  // Normalize vertical height

        float h_k = sqrt(deltaX * deltaX + deltaY * deltaY);

        // Calculate ur, uz based on h_k and z_k
        float ur = (h_k < 30 && z_k < 60) ? u_r[cell_cent] : 0;
        float uz = (h_k < 30 && z_k < 60) ? u_z[cell_cent] : 0;

        // Compute potential velocities
        float u_p = U_c * ur * (deltaX / h_k);
        float v_p = U_c * ur * (deltaY / h_k);
        float w_p = U_c * uz;

        // Accumulate potentials
        atomicAdd(&Pot_u[cell_cent], u_p);
        atomicAdd(&Pot_v[cell_cent], v_p);
        atomicAdd(&Pot_w[cell_cent], w_p);
    }
}


// Main function
void Fire::potentialGlobal(WINDSGeneralData* WGD) {
    const int gridSize = (nx - 1) * (ny - 1) * (nz - 1);

    // Allocate and initialize device memory
    thrust::device_vector<float> d_Pot_u(gridSize, 0.0f);
    thrust::device_vector<float> d_Pot_v(gridSize, 0.0f);
    thrust::device_vector<float> d_Pot_w(gridSize, 0.0f);

    // Allocate and transfer `burn_flag` and `fire_cells` to device
    thrust::device_vector<int> d_burn_flag(burn_flag, burn_flag + (nx - 1) * (ny - 1));
    thrust::device_vector<float> d_H0_array(H0_array, H0_array + (nx - 1) * (ny - 1));

    // Allocate temporary variables for reduction (heat release)
    float* d_H0_sum;
    int* d_icent, * d_jcent, * d_counter;
    CUDA_CHECK(cudaMalloc(&d_H0_sum, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_icent, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_jcent, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));

    // Initialize these on the device
    CUDA_CHECK(cudaMemset(d_H0_sum, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_icent, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_jcent, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    dim3 threads(16, 16);
    dim3 grid((nx + threads.x - 1) / threads.x, (ny + threads.y - 1) / threads.y);

    // Launch heat release computation kernel
    compute_heat_release<<<grid, threads>>>(
        nx, ny,
        thrust::raw_pointer_cast(d_burn_flag.data()),
        thrust::raw_pointer_cast(d_H0_array.data()),
        dx, dy, d_H0_sum, d_icent, d_jcent, d_counter
    );

    // Copy results back for host processing (if needed)
    float H0_host;
    CUDA_CHECK(cudaMemcpy(&H0_host, d_H0_sum, sizeof(float), cudaMemcpyDeviceToHost));

    if (H0_host > 0) {
        // Launch potential field computation kernel
        dim3 threads3D(8, 8, 8);
        dim3 grid3D((nx + threads3D.x - 1) / threads3D.x, 
                    (ny + threads3D.y - 1) / threads3D.y, 
                    (nz + threads3D.z - 1) / threads3D.z);

        calculate_potential<<<grid3D, threads3D>>>(
            thrust::raw_pointer_cast(d_Pot_u.data()),
            thrust::raw_pointer_cast(d_Pot_v.data()),
            thrust::raw_pointer_cast(d_Pot_w.data()),
            thrust::raw_pointer_cast(d_z_mix.data()),
            nx, ny, nz, dx, dy, dz, /* Other parameters */
            U_c, L_c
        );
    }

    // Ensure CUDA kernel execution has completed
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the results back to host arrays if needed
    thrust::copy(d_Pot_u.begin(), d_Pot_u.end(), Pot_u.begin());
    thrust::copy(d_Pot_v.begin(), d_Pot_v.end(), Pot_v.begin());
    thrust::copy(d_Pot_w.begin(), d_Pot_w.end(), Pot_w.begin());

    // Free temporary device memory
    CUDA_CHECK(cudaFree(d_H0_sum));
    CUDA_CHECK(cudaFree(d_icent));
    CUDA_CHECK(cudaFree(d_jcent));
    CUDA_CHECK(cudaFree(d_counter));
}