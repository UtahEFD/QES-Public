/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Matthew Moody
 * Copyright (c) 2024 Jeremy Gibbs
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Brian Bailey
 * Copyright (c) 2024 Pete Willemsen
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

# include "Fire.h"

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
	int nx, int ny,
	int i, int j,
	float* d_front_map, float* d_fire_cells
) {
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	int jj = blockDim.y * blockIdx.y + threadIdx.y;

	if (ii >= nx - 1 || jj >= ny - 1) return;

	float sdf = 1000;
	float sdf_min;
	int idx = i + j * (nx - 1);
	int idx2 = ii + jj * (nx - 1);

 	sdf = 1000;
 
    if (d_fire_cells[idx2] == 1) {
        sdf_min = sqrt((ii - i) * (ii - i) + (jj - j) * (jj - j));
    } else {
        sdf_min = 1000;
	}
    sdf = sdf_min < sdf ? sdf_min : sdf;
	d_front_map[idx] = sdf;
}

void Fire ::LSinitGlob()
{
    auto start = std::chrono::high_resolution_clock::now();// Start recording execution time
    float sdf, sdf_min;
    /**
     * Set up initial level set using signed distance function.
     */    
    for (int j = 0; j < ny - 1; j++) {
        for (int i = 0; i < nx - 1; i++) {
	        int idx = i + j * (nx - 1);
	        if (fire_cells[idx].state.front_flag == 1) {
	            front_map[idx] = 0;
	        } else {

				// CUDA CALL HERE!
	            sdf = 1000;
    	        for (int jj = 0; jj < ny - 1; jj++) {
	                for (int ii = 0; ii < nx - 1; ii++) {
	                    int idx2 = ii + jj * (nx - 1);
	                    if (fire_cells[idx2].state.front_flag == 1) {
		                    sdf_min = sqrt((ii - i) * (ii - i) + (jj - j) * (jj - j));
	                    } else {
		                    sdf_min = 1000;
	                    }
	                    sdf = sdf_min < sdf ? sdf_min : sdf;
	                }
	            }   
	            front_map[idx] = sdf;
				//CUDA CALL HERE 
	        }
        }
    }
    auto finish = std::chrono::high_resolution_clock::now();// Finish recording execution time
    std::chrono::duration<float> elapsed = finish - start;
    std::cout << "[QES-Fire] LS init elapsed time:\t" << elapsed.count() << " s\n";// Print out elapsed execution time
}