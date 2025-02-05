/****************************************************************************
 * Copyright (c) 2025 University of Utah
 * Copyright (c) 2025 University of Minnesota Duluth
 *
 * Copyright (c) 2025 Matthew Moody
 * Copyright (c) 2025 Jeremy Gibbs
 * Copyright (c) 2025 Rob Stoll
 * Copyright (c) 2025 Fabien Margairaz
 * Copyright (c) 2025 Brian Bailey
 * Copyright (c) 2025 Pete Willemsen
 *
 * This file is part of QES-Fire
 *
 * GPL-3.0 License
 *
 * QES-Fire is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Fire is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/
/**
 * @file LSinitGlob.cu
 * @brief This file initializes the level set using CUDA
 */

# include "../Fire.h"
# include "LSinitGlob.h"

using namespace std;

#define cudaCheck(x) _cudaCheck(x, #x, __FILE__, __LINE__)

template<typename T>
void _cudaCheck(T e, const char *func, const char *call, const int line)
{
  if (e != cudaSuccess) {
    printf("\"%s\" at %d in %s\n\treturned %d\n-> %s\n", func, line, call, (int)e, cudaGetErrorString(e));
    exit(EXIT_FAILURE);
  }
}

__global__
void LSGlob(
	int nx, int ny,	float* d_front_map
) {
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	int jj = blockDim.y * blockIdx.y + threadIdx.y;

	if (ii >= nx - 1 || jj >= ny - 1) return;
	int idx = ii + jj * (nx - 1);

	if (d_front_map[idx] == 0) return;
	float sdf = 1000;
	float sdf_min = 1000;
 	for (int j = 0; j < ny - 1; j++) {
		for (int i = 0; i < nx - 1; i++) {
	        int idx2 = i + j * (nx - 1);
	        if (d_front_map[idx2] == 0) {
	            sdf_min = sqrtf((ii - i) * (ii - i) + (jj - j) * (jj - j));
				sdf = sdf_min < sdf ? sdf_min : sdf;
	        }
		} 
	}
	d_front_map[idx] = sdf;
}

void Fire ::LSinitGlob()
{
    auto start = std::chrono::high_resolution_clock::now();// Start recording execution time
	// Initialize level set map at 1000
	std::fill(front_map.begin(), front_map.end(), 1000.0);
	int gridSize = (nx - 1) * (ny - 1);
    // If a cell is burning, set level set to 0    
    for (int j = 0; j < ny - 1; j++) {
        for (int i = 0; i < nx - 1; i++) {
	        int idx = i + j * (nx - 1);
	        if (fire_cells[idx].state.front_flag == 1) {
	            front_map[idx] = 0;
	        }
		}
	}

	//allocate and initialize LS map
    cudaMalloc((void **)&d_front_map, gridSize * sizeof(float));
	cudaMemcpy(d_front_map, front_map.data(), gridSize * sizeof(float), cudaMemcpyHostToDevice);
	
	
	// Call LSGlob kernal
	dim3 threadsPerBlock(32,32);
    dim3 numBlocks(ceil(nx/16), ceil(ny/16));
	LSGlob<<<numBlocks, threadsPerBlock>>>(nx, ny, d_front_map);

	cudaCheck(cudaGetLastError());  
	// Copy data from device to host
	cudaMemcpy(front_map.data(), d_front_map, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
	// Free memory	
	cudaFree(d_front_map);

    auto finish = std::chrono::high_resolution_clock::now();// Finish recording execution time
    std::chrono::duration<float> elapsed = finish - start;
    std::cout << "[QES-Fire] LS init GPU elapsed time:\t" << elapsed.count() << " s\n";// Print out elapsed execution time
}