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

  auto start = std::chrono::high_resolution_clock::now();

  int id, icell_face, icell_cent;
  WRFInput *wrf_ptr = WID->simParams->wrfInputData;

  // Create initial wind field in the area WRF data is available
  for (auto i = WGD->halo_index_x; i < WGD->wrf_nx + WGD->halo_index_x - 1; i++) {
    for (auto j = WGD->halo_index_y; j < WGD->wrf_ny + WGD->halo_index_y - 1; j++) {
      id = (i - WGD->halo_index_x) + (j - WGD->halo_index_y) * (WGD->wrf_nx);
      for (auto k = 1; k < WGD->nz; k++) {
        icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
        WGD->u0[icell_face] = 0.5 * (wrf_ptr->u0_fmw[icell_cent] + wrf_ptr->u0_fmw[icell_cent + 1]);
        WGD->v0[icell_face] = 0.5 * (wrf_ptr->v0_fmw[icell_cent] + wrf_ptr->v0_fmw[icell_cent + (WGD->nx - 1)]);
      }
    }
  }

  for (auto i = WGD->halo_index_x + 1; i < WGD->wrf_nx + WGD->halo_index_x; i++) {
    for (auto j = 0; j < WGD->halo_index_y + 1; j++) {
      for (auto k = 1; k < WGD->nz; k++) {
        icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        id = i + (WGD->halo_index_y + 1) * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->u0[icell_face] = WGD->u0[id];
        WGD->v0[icell_face] = WGD->v0[id];
      }
    }

    for (auto j = WGD->wrf_ny + WGD->halo_index_y; j < WGD->ny; j++) {
      for (auto k = 1; k < WGD->nz; k++) {
        icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        id = i + (WGD->wrf_ny + WGD->halo_index_y - 1) * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->u0[icell_face] = WGD->u0[id];
        WGD->v0[icell_face] = WGD->v0[id];
      }
    }
  }

  for (auto j = WGD->halo_index_y + 1; j < WGD->wrf_ny + WGD->halo_index_y; j++) {
    for (auto i = 0; i < WGD->halo_index_x + 1; i++) {
      for (auto k = 1; k < WGD->nz; k++) {
        icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        id = WGD->halo_index_x + 1 + j * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->u0[icell_face] = WGD->u0[id];
        WGD->v0[icell_face] = WGD->v0[id];
      }
    }

    for (auto i = WGD->wrf_nx + WGD->halo_index_x; i < WGD->nx; i++) {
      for (auto k = 1; k < WGD->nz; k++) {
        icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        id = (WGD->wrf_nx + WGD->halo_index_x - 1) + j * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->u0[icell_face] = WGD->u0[id];
        WGD->v0[icell_face] = WGD->v0[id];
      }
    }
  }

  for (auto i = 0; i < WGD->halo_index_x + 1; i++) {
    for (auto j = 0; j < WGD->halo_index_y + 1; j++) {
      for (auto k = 1; k < WGD->nz; k++) {
        icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        id = WGD->halo_index_x + 1 + (WGD->halo_index_y + 1) * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->u0[icell_face] = WGD->u0[id];
        WGD->v0[icell_face] = WGD->v0[id];
      }
    }

    for (auto j = WGD->wrf_ny + WGD->halo_index_y; j < WGD->ny; j++) {
      for (auto k = 1; k < WGD->nz; k++) {
        icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        id = WGD->halo_index_x + 1 + (WGD->wrf_ny + WGD->halo_index_y - 1) * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->u0[icell_face] = WGD->u0[id];
        WGD->v0[icell_face] = WGD->v0[id];
      }
    }
  }

  for (auto i = WGD->wrf_nx + WGD->halo_index_x; i < WGD->halo_index_x + 1; i++) {
    for (auto j = 0; j < WGD->halo_index_y + 1; j++) {
      for (auto k = 1; k < WGD->nz; k++) {
        icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        id = WGD->wrf_nx + WGD->halo_index_x - 1 + (WGD->halo_index_y + 1) * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->u0[icell_face] = WGD->u0[id];
        WGD->v0[icell_face] = WGD->v0[id];
      }
    }

    for (auto j = WGD->wrf_ny + WGD->halo_index_y; j < WGD->ny; j++) {
      for (auto k = 1; k < WGD->nz; k++) {
        icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        id = WGD->wrf_nx + WGD->halo_index_x - 1 + (WGD->wrf_ny + WGD->halo_index_y - 1) * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->u0[icell_face] = WGD->u0[id];
        WGD->v0[icell_face] = WGD->v0[id];
      }
    }
  }

  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> elapsed = finish - start;
  //std::cout << "Elapsed time for Barnes interpolation on CPU: " << elapsed.count() << " s\n";

  return;
}
