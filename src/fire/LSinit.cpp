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
 * @file LSinit.cpp
 * @brief This file initializes the level set
 */

# include "Fire.h"

using namespace std;

void Fire ::LSinit()
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
	        }
        }
    }
    auto finish = std::chrono::high_resolution_clock::now();// Finish recording execution time
    std::chrono::duration<float> elapsed = finish - start;
    std::cout << "[QES-Fire] LS init elapsed time:\t" << elapsed.count() << " s\n";// Print out elapsed execution time
}