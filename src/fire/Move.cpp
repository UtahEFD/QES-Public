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
 * @file Move.cpp
 * @brief This function advances the level set
 */
#include "Fire.h"

void Fire ::move(WINDSGeneralData *WGD)
{
  std::cout << "[QES-Fire]\t Advancing level set..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();// Start recording execution time

  auto [nx, ny, nz] = WGD->domain.getDomainCellNum();

  // compute time step
  dt = computeTimeStep();

  if (FFII_flag == 1) {
    int FT_idx1 = 0;
    float currTime = time;// calculate simTimeCurrent
    float nextTime = currTime + dt;
    // Find all ignition times during current timestep and ignite in domain if not burned
    for (int it = 0; it < FT_time.size(); it++) {
      if (FT_time[it] >= currTime && FT_time[it] <= nextTime) {
        int nx1 = round(FT_x1[it] / dx);
        if (nx1 > nx - 1) {
          nx1 = nx - 1;
        }
        int ny1 = round((750 - FT_y1[it]) / dy);
        if (ny1 > ny - 1) {
          ny1 = ny - 1;
        }
        FT_idx1 = nx1 + ny1 * (nx - 1);
        if (burn_flag[FT_idx1] < 2) {
          front_map[FT_idx1] = 0;
          fire_cells[FT_idx1].state.burn_flag = 1;
        }
      }
    }
  }
  for (int j = 1; j < ny - 2; j++) {
    for (int i = 1; i < nx - 2; i++) {
      int idx = i + j * (nx - 1);
      // get fire properties at this location
      struct FireProperties fp = fire_cells[idx].properties;
      struct FuelProperties *fuel = fire_cells[idx].fuel;
      float H = fire_cells[idx].properties.h * (1 - (fire_cells[idx].state.burn_time / fire_cells[idx].properties.tau));
      float maxH = fire_cells[idx].properties.h;
      float T = WGD->terrain[idx];
      float D = fuel->fuelDepth * 0.3048;
      int TID = std::round(T / dz);
      float FD = H / 2.0 + T + D;
      float MFD = maxH + T + D;
      int kh = 0;
      int maxkh = 0;

      if (H == 0) {
        kh = std::round(T / dz);
      } else {
        kh = std::round(FD / dz);
      }
      if (maxH == 0) {
        maxkh = std::round(T / dz);
      } else {
        maxkh = std::round(MFD / dz);
      }
      // if burn flag = 1, update burn time
      if (burn_flag[idx] == 1) {
        fire_cells[idx].state.burn_time += dt;
      }
      // set burn flag to 2 (burned) if residence time exceeded, set Forcing function to 0, and update z0 to bare soil
      if (fire_cells[idx].state.burn_time >= fp.tau && fire_cells[idx].state.burn_flag == 1) {
        fire_cells[idx].state.burn_flag = 2;
        Force[idx] = 0;
        H0[idx] = 0;
        for (int k = TID; k <= maxkh; k++) {
          int icell_cent = i + j * (nx - 1) + (k) * (nx - 1) * (ny - 1);
          WGD->icellflag[icell_cent] = 1;
        }
        // Need to fix where z0 is reset MM
        // WGD->z0_domain[idx] = 0.01;
      }
      // advance level set
      front_map[idx] = front_map[idx] - dt * (fmax(Force[idx], 0) * del_plus[idx] + fmin(Force[idx], 0) * del_min[idx]);
      // if level set <= 1, set burn_flag to 0.5 - L.S. for preheating
      if (front_map[idx] <= 1 && burn_flag[idx] < 1) {
        fire_cells[idx].state.burn_flag = 0.5;
      }
      // if level set < threshold, set burn flag to 1 and start smoke flag
      if (front_map[idx] <= 0.1 && burn_flag[idx] < 1) {
        fire_cells[idx].state.burn_flag = 1;
        smoke_flag[idx] = 1;
      }
      // update burn flag field
      burn_flag[idx] = fire_cells[idx].state.burn_flag;
      burn_out[idx] = burn_flag[idx];
    }
  }

  // advance time
  time += dt;
  auto finish = std::chrono::high_resolution_clock::now();// Finish recording execution time
  std::chrono::duration<float> elapsed = finish - start;
  std::cout << "\t\t elapsed time:\t" << elapsed.count() << " s" << std::endl;// Print out elapsed execution time
}
