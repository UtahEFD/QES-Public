/****************************************************************************
 * Copyright (c) 2021 University of Utah
 * Copyright (c) 2021 University of Minnesota Duluth
 *
 * Copyright (c) 2021 Behnam Bozorgmehr
 * Copyright (c) 2021 Jeremy A. Gibbs
 * Copyright (c) 2021 Fabien Margairaz
 * Copyright (c) 2021 Eric R. Pardyjak
 * Copyright (c) 2021 Zachary Patterson
 * Copyright (c) 2021 Rob Stoll
 * Copyright (c) 2021 Pete Willemsen
 *
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/**
 * @file Sensor.cpp
 * @brief Collection of variables containing information relevant to
 * sensors read from an xml.
 *
 * @sa ParseInterface
 * @sa TimeSeries
 */

#include <math.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <limits>

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"
#include "WindProfilerWRF.h"


void WindProfilerWRF::interpolateWindProfile(const WINDSInputData *WID, WINDSGeneralData *WGD)
{
  std::cout << "Processing WRF u0_fmw, v0_fmw and w0_fmw into initial QES wind fields." << std::endl;

  /* Need linear interoplation because
     WGD->u0 and WGD->v0 are on faces (size nx*ny*nz)
     ----
     wrf_ptr->u0_fmw and wrf_ptr->v0_fmw are size wrf_ptr->fm_nx * wrf_ptr->fm_ny * wrf_ptr->ht_fmw.size()
     the heights themselves are in the wrf_ptr->ht_fmw array
  */

  auto start = std::chrono::high_resolution_clock::now();

  int id, icell_face, icell_cent;
  float zm;
  WRFInput *wrf_ptr = WID->simParams->wrfInputData;

  // Create initial wind field in the area WRF data is available
  for (auto k = 1; k < WGD->nz - 1; ++k) {
    for (auto j = WGD->halo_index_y; j < WGD->wrf_ny + WGD->halo_index_y; ++j) {
      for (auto i = WGD->halo_index_x + 1; i < WGD->wrf_nx + WGD->halo_index_x; ++i) {
        id = (i - WGD->halo_index_x) + (j - WGD->halo_index_y) * WGD->wrf_nx;
        zm = WGD->z[k] - WGD->z_face[WGD->terrain_face_id[i + j * WGD->nx] - 1];
        icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        interpolate(WGD->u0[icell_face], zm, id, wrf_ptr->u0_fmw, wrf_ptr->ht_fmw, 1, WGD->wrf_nx * WGD->wrf_ny);
      }
    }
  }
  for (auto k = 1; k < WGD->nz - 1; ++k) {
    for (auto j = WGD->halo_index_y + 1; j < WGD->wrf_ny + WGD->halo_index_y; ++j) {
      for (auto i = WGD->halo_index_x; i < WGD->wrf_nx + WGD->halo_index_x; ++i) {
        id = (i - WGD->halo_index_x) + (j - WGD->halo_index_y) * WGD->wrf_nx;
        zm = WGD->z[k] - WGD->z_face[WGD->terrain_face_id[i + j * WGD->nx] - 1];
        icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        interpolate(WGD->v0[icell_face], zm, id, wrf_ptr->v0_fmw, wrf_ptr->ht_fmw, WGD->wrf_nx, WGD->wrf_nx * WGD->wrf_ny);
      }
    }
  }

  // top ghost cell (above domain)
  for (auto j = WGD->halo_index_y + 1; j < WGD->wrf_ny + WGD->halo_index_y; ++j) {
    for (auto i = WGD->halo_index_x + 1; i < WGD->wrf_nx + WGD->halo_index_x; ++i) {
      int icell_face_up = i + j * WGD->nx + (WGD->nz - 1) * WGD->nx * WGD->ny;
      int icell_face_down = i + j * WGD->nx + (WGD->nz - 2) * WGD->nx * WGD->ny;
      WGD->u0[icell_face_up] = WGD->u0[icell_face_down];
      WGD->v0[icell_face_up] = WGD->v0[icell_face_down];
    }
  }
  // bottom ghost cell (under the terrain)
  for (auto j = 0; j < WGD->ny; ++j) {
    for (auto i = 0; i < WGD->nx; ++i) {
      icell_face = i + j * WGD->nx;
      WGD->u0[icell_face] = 0.0;
      WGD->v0[icell_face] = 0.0;
    }
  }

  // sides (including halo cells)
  // u-velocity
  for (auto k = 1; k < WGD->nz; ++k) {
    // West side (inside domain only)
    for (auto j = WGD->halo_index_y + 1; j < WGD->wrf_ny + WGD->halo_index_y; ++j) {
      for (auto i = 0; i <= WGD->halo_index_x; ++i) {
        icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        id = (WGD->halo_index_x + 1) + j * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->u0[icell_face] = WGD->u0[id];
      }
    }
    // East side (inside domain only)
    for (auto j = WGD->halo_index_y + 1; j < WGD->wrf_ny + WGD->halo_index_y; ++j) {
      for (auto i = WGD->wrf_ny + WGD->halo_index_y; i < WGD->nx; ++i) {
        icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        id = (WGD->wrf_ny + WGD->halo_index_x - 1) + j * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->u0[icell_face] = WGD->u0[id];
      }
    }
    // South side (whole domain)
    for (auto j = 0; j <= WGD->halo_index_y; ++j) {
      for (auto i = 0; i < WGD->nx; ++i) {
        icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        id = i + (WGD->halo_index_y + 1) * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->u0[icell_face] = WGD->u0[id];
      }
    }
    // North side (whole domain)
    for (auto j = WGD->wrf_nx + WGD->halo_index_x; j < WGD->ny; ++j) {
      for (auto i = 0; i < WGD->nx; ++i) {
        icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        id = i + (WGD->wrf_ny + WGD->halo_index_y - 1) * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->u0[icell_face] = WGD->u0[id];
      }
    }
  }

  // v-velocity
  for (auto k = 1; k < WGD->nz; ++k) {
    // South side (inside domain only)
    for (auto j = 0; j <= WGD->halo_index_y; ++j) {
      for (auto i = WGD->halo_index_x + 1; i < WGD->wrf_nx + WGD->halo_index_x; ++i) {
        icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        id = i + (WGD->halo_index_y + 1) * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->v0[icell_face] = WGD->v0[id];
      }
    }
    // North side (inside domain only)
    for (auto j = WGD->wrf_nx + WGD->halo_index_y; j < WGD->ny; ++j) {
      for (auto i = WGD->halo_index_x + 1; i < WGD->wrf_nx + WGD->halo_index_x; ++i) {
        icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        id = i + (WGD->wrf_nx + WGD->halo_index_x - 1) * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->v0[icell_face] = WGD->v0[id];
      }
    }
    // West side (whole domain)
    for (auto j = 0; j < WGD->ny; ++j) {
      for (auto i = 0; i <= WGD->halo_index_x; ++i) {
        icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        id = (WGD->halo_index_x + 1) + j * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->v0[icell_face] = WGD->v0[id];
      }
    }
    // East side (whole domain)
    for (auto j = 0; j < WGD->ny; ++j) {
      for (auto i = WGD->wrf_nx + WGD->halo_index_x; i < WGD->nx; ++i) {
        icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        id = (WGD->wrf_ny + WGD->halo_index_x - 1) + j * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->v0[icell_face] = WGD->v0[id];
      }
    }
  }

  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> elapsed = finish - start;
  //std::cout << "Elapsed time for Barnes interpolation on CPU: " << elapsed.count() << " s\n";

  return;
}


void WindProfilerWRF::interpolate(float &u, float &zm, int &id, std::vector<float> &u_fmw, std::vector<float> &z_fmw, int s1, int s2)
{
  if (zm < 0) {
    // if zm is under the surface -> u = 0
    u = 0.0;
  } else if (zm <= z_fmw[0]) {
    // if zm is below the first height -> linear interp with 0 at surface
    // !could be replaced by log-law if z0 is provided
    u = 0.5 * (zm * (u_fmw[id]) / (z_fmw[0]) + zm * (u_fmw[id - s1]) / (z_fmw[0]));
  } else if (zm >= z_fmw.back()) {
    // if zm is above top last height -> constant u with last node
    u = 0.5 * (u_fmw[id + z_fmw.size() * s2] + u_fmw[id - s1 + z_fmw.size() * s2]);
  } else {
    int kl, kt;
    kl = lower_bound(z_fmw.begin(), z_fmw.end(), zm) - z_fmw.begin();
    kt = kl + 1;
    // vertical interpolation at cell head of face
    u = (u_fmw[id + kl * s2] + (zm - z_fmw[kl]) * (u_fmw[id + kt * s2] - u_fmw[id + kl * s2]) / (z_fmw[kt] - z_fmw[kl]));
    //vertical interpolation at cell behind of face
    u += (u_fmw[id - s1 + kl * s2] + (zm - z_fmw[kl]) * (u_fmw[id - s1 + kt * s2] - u_fmw[id - s1 + kl * s2]) / (z_fmw[kt] - z_fmw[kl]));
    // averaging....
    u *= 0.5;
  }

  return;
}
