/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Behnam Bozorgmehr
 * Copyright (c) 2024 Jeremy A. Gibbs
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Eric R. Pardyjak
 * Copyright (c) 2024 Zachary Patterson
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Lucas Ulmer
 * Copyright (c) 2024 Pete Willemsen
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
#include "WindProfilerBarnCPU.h"


void WindProfilerBarnCPU::interpolateWindProfile(const WINDSInputData *WID, WINDSGeneralData *WGD)
{
  sensorsProfiles(WID, WGD);
  int num_sites = available_sensor_id.size();
  
  if (num_sites == 1) {
    singleSensorInterpolation(WGD);
  } else {
    // If number of sites are more than one
    // Apply 2D Barnes scheme to interpolate site velocity profiles to the whole domain
    auto startBarnesCPU = std::chrono::high_resolution_clock::now();
    BarnesInterpolationCPU(WID, WGD);
    auto finishBarnesCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsedBarnesCPU = finishBarnesCPU - startBarnesCPU;
    std::cout << "Elapsed time for Barnes interpolation on CPU: " << elapsedBarnesCPU.count() << " s\n";
  }
  return;
}

void WindProfilerBarnCPU::BarnesInterpolationCPU(const WINDSInputData *WID, WINDSGeneralData *WGD)
{
  auto [nx, ny, nz] = WGD->domain.getDomainCellNum();
  auto [dx, dy, dz] = WGD->domain.getDomainSize();

  std::vector<float> x, y;
  x.resize(nx);
  for (auto i = 0; i < nx; i++) {
    x[i] = (i - 0.5) * dx; /**< Location of face centers in x-dir */
  }

  y.resize(ny);
  for (auto j = 0; j < ny; j++) {
    y[j] = (j - 0.5) * dy; /**< Location of face centers in y-dir */
  }

  float rc_sum, rc_val, xc, yc, rc;
  float dn, lamda, s_gamma;
  float sum_wm, sum_wu, sum_wv;
  float dxx, dyy, u12, u34, v12, v34;
  int icell_face;
  float z_terrain;

  int num_sites = available_sensor_id.size();

  std::vector<float> u0_int(num_sites * nz, 0.0);
  std::vector<float> v0_int(num_sites * nz, 0.0);
  std::vector<std::vector<std::vector<float>>> wm(num_sites, std::vector<std::vector<float>>(nx, std::vector<float>(ny, 0.0)));
  std::vector<std::vector<std::vector<float>>> wms(num_sites, std::vector<std::vector<float>>(nx, std::vector<float>(ny, 0.0)));
  int iwork = 0, jwork = 0;
  std::vector<int> site_i(num_sites, 0);
  std::vector<int> site_j(num_sites, 0);
  std::vector<int> site_id(num_sites, 0);
 
  
  rc_sum = 0.0;
  for (auto i = 0; i < num_sites; i++) {
    rc_val = 1000000.0;
    for (auto ii = 0; ii < num_sites; ii++) {
      xc = WID->metParams->sensors[available_sensor_id[ii]]->site_xcoord - WID->metParams->sensors[available_sensor_id[i]]->site_xcoord;
      yc = WID->metParams->sensors[available_sensor_id[ii]]->site_ycoord - WID->metParams->sensors[available_sensor_id[i]]->site_ycoord;
      rc = sqrt(pow(xc, 2.0) + pow(yc, 2.0));
      if (rc < rc_val && ii != i) {
        rc_val = rc;
      }
    }
    rc_sum = rc_sum + rc_val;
  }

  dn = rc_sum / num_sites;
  lamda = 5.052 * pow((2 * dn / M_PI), 2.0);
  s_gamma = 0.2;
  for (auto j = 0; j < ny; j++) {
    for (auto i = 0; i < nx; i++) {
      sum_wm = 0.0;
      for (auto ii = 0; ii < num_sites; ii++) {
        wm[ii][i][j] = exp((-1 / lamda)
                             * pow(WID->metParams->sensors[available_sensor_id[ii]]->site_xcoord - x[i], 2.0)
                           - (1 / lamda) * pow(WID->metParams->sensors[available_sensor_id[ii]]->site_ycoord - y[j], 2.0));
        wms[ii][i][j] = exp((-1 / (s_gamma * lamda))
                              * pow(WID->metParams->sensors[available_sensor_id[ii]]->site_xcoord - x[i], 2.0)
                            - (1 / (s_gamma * lamda)) * pow(WID->metParams->sensors[available_sensor_id[ii]]->site_ycoord - y[j], 2.0));
        sum_wm += wm[ii][i][j];
      }
      if (sum_wm == 0) {
        for (auto ii = 0; ii < num_sites; ii++) {
          wm[ii][i][j] = 1e-20;
        }
      }
    }
  }

  int k_mod;// Modified index in z-direction
  for (auto k = 0; k < nz - 1; k++) {
    for (auto j = 0; j < ny; j++) {
      for (auto i = 0; i < nx; i++) {
        sum_wu = 0.0;
        sum_wv = 0.0;
        sum_wm = 0.0;
        int id = i + j * nx;// Index in horizontal surface
	z_terrain = WGD->domain.z_face[WGD->terrain_face_id[id]];
        // If height added to top of terrain is still inside QES domain
        if (k + WGD->terrain_face_id[id] < nz) {
          k_mod = k + WGD->terrain_face_id[id];// Set the modified index
        } else {
          continue;
        }

        for (auto ii = 0; ii < num_sites; ii++) {
          site_i[ii] = WID->metParams->sensors[available_sensor_id[ii]]->site_xcoord / dx;
          site_j[ii] = WID->metParams->sensors[available_sensor_id[ii]]->site_ycoord / dy;
          site_id[ii] = site_i[ii] + site_j[ii] * nx;
          // If the height difference between the terrain at the curent cell and sensor location is less than ABL height
          if (abs(WGD->domain.z[WGD->terrain_face_id[id]] - WGD->domain.z[WGD->terrain_face_id[site_id[ii]]]) > abl_height[ii]) {
            surf_layer_height = asl_percent * abl_height[ii];
          } else {
            surf_layer_height = asl_percent * (2 * abl_height[ii] - abs(WGD->domain.z[WGD->terrain_face_id[id]] - WGD->domain.z[WGD->terrain_face_id[site_id[ii]]]));
          }
          // If sum of z index and the terrain index at the sensor location is outside the domain
          if (k + WGD->terrain_face_id[site_id[ii]] > nz - 2) {
            sum_wu += wm[ii][i][j] * u_prof[ii * nz + nz - 2];
            sum_wv += wm[ii][i][j] * v_prof[ii * nz + nz - 2];
            sum_wm += wm[ii][i][j];
          }// If height (above ground) is less than or equal to ASL height
          else if ( (WGD->domain.z[k_mod]-z_terrain) <= surf_layer_height) {
            sum_wu += wm[ii][i][j] * u_prof[ii * nz + k + WGD->terrain_face_id[site_id[ii]]];
            sum_wv += wm[ii][i][j] * v_prof[ii * nz + k + WGD->terrain_face_id[site_id[ii]]];
            sum_wm += wm[ii][i][j];
          }// If height (above ground) is greater than ASL height and modified index is inside the domain
          else if ((WGD->domain.z[k_mod]-z_terrain) > surf_layer_height
                   && k + WGD->terrain_face_id[site_id[ii]] < nz
                   && k_mod > k + WGD->terrain_face_id[site_id[ii]]) {
            sum_wu += wm[ii][i][j] * u_prof[ii * nz + k_mod];
            sum_wv += wm[ii][i][j] * v_prof[ii * nz + k_mod];
            sum_wm += wm[ii][i][j];
	    } else {
            sum_wu += wm[ii][i][j] * u_prof[ii * nz + k + WGD->terrain_face_id[site_id[ii]]];
            sum_wv += wm[ii][i][j] * v_prof[ii * nz + k + WGD->terrain_face_id[site_id[ii]]];
            sum_wm += wm[ii][i][j];
          }
	  
        }

        icell_face = i + j * nx + k_mod * nx * ny;
        WGD->u0[icell_face] = sum_wu / sum_wm;
        WGD->v0[icell_face] = sum_wv / sum_wm;
        WGD->w0[icell_face] = 0.0;
      }
    }
  }

  for (auto ii = 0; ii < num_sites; ii++) {
    if (WID->metParams->sensors[available_sensor_id[ii]]->site_xcoord > 0
        && WID->metParams->sensors[available_sensor_id[ii]]->site_xcoord < (nx - 1) * dx
        && WID->metParams->sensors[available_sensor_id[ii]]->site_ycoord > 0
        && WID->metParams->sensors[available_sensor_id[ii]]->site_ycoord < (ny - 1) * dy) {
      for (auto j = 0; j < ny; j++) {
        if (y[j] < WID->metParams->sensors[available_sensor_id[ii]]->site_ycoord) {
          jwork = j;
        }
      }

      for (auto i = 0; i < nx; i++) {
        if (x[i] < WID->metParams->sensors[available_sensor_id[ii]]->site_xcoord) {
          iwork = i;
        }
      }

      int id = iwork + jwork * nx;
      for (auto k_mod = WGD->terrain_face_id[id]; k_mod < nz; k_mod++) {
        dxx = WID->metParams->sensors[available_sensor_id[ii]]->site_xcoord - x[iwork];
        dyy = WID->metParams->sensors[available_sensor_id[ii]]->site_ycoord - y[jwork];
        int index_work = iwork + jwork * nx + k_mod * nx * ny;
        u12 = (1 - (dxx / dx)) * WGD->u0[index_work + nx] + (dxx / dx) * WGD->u0[index_work + 1 + nx];
        u34 = (1 - (dxx / dx)) * WGD->u0[index_work] + (dxx / dx) * WGD->u0[index_work + 1];
        u0_int[k_mod + ii * nz] = (dyy / dy) * u12 + (1 - (dyy / dy)) * u34;

        v12 = (1 - (dxx / dx)) * WGD->v0[index_work + nx] + (dxx / dx) * WGD->v0[index_work + 1 + nx];
        v34 = (1 - (dxx / dx)) * WGD->v0[index_work] + (dxx / dx) * WGD->v0[index_work + 1];
        v0_int[k_mod + ii * nz] = (dyy / dy) * v12 + (1 - (dyy / dy)) * v34;
      }
    } else {
      site_i[ii] = WID->metParams->sensors[available_sensor_id[ii]]->site_xcoord / dx;
      site_j[ii] = WID->metParams->sensors[available_sensor_id[ii]]->site_ycoord / dy;
      site_id[ii] = site_i[ii] + site_j[ii] * nx;
      for (auto k = 0; k < nz; k++) {
        if (k + WGD->terrain_face_id[site_id[ii]] > nz - 2) {
          u0_int[k + ii * nz] = u_prof[ii * nz + nz - 2];
          v0_int[k + ii * nz] = v_prof[ii * nz + nz - 2];
        } else {
          u0_int[k + WGD->terrain_face_id[site_id[ii]] + ii * nz] = u_prof[ii * nz + k + WGD->terrain_face_id[site_id[ii]]];
          v0_int[k + WGD->terrain_face_id[site_id[ii]] + ii * nz] = v_prof[ii * nz + k + WGD->terrain_face_id[site_id[ii]]];
        }
      }
    }
  }


  for (auto k = 0; k < nz - 1; k++) {
    for (auto j = 0; j < ny; j++) {
      for (auto i = 0; i < nx; i++) {
        sum_wu = 0.0;
        sum_wv = 0.0;
        sum_wm = 0.0;
        int id = i + j * nx;//Index in horizontal surface
	z_terrain = WGD->domain.z_face[WGD->terrain_face_id[id]];
        //If height added to top of terrain is still inside QES domain
        if (k + WGD->terrain_face_id[id] < nz) {
          k_mod = k + WGD->terrain_face_id[id];//Set the modified index
        } else {
          continue;
        }
        for (auto ii = 0; ii < num_sites; ii++) {
          site_i[ii] = WID->metParams->sensors[available_sensor_id[ii]]->site_xcoord / dx;
          site_j[ii] = WID->metParams->sensors[available_sensor_id[ii]]->site_ycoord / dy;
          site_id[ii] = site_i[ii] + site_j[ii] * nx;
          // If the height difference between the terrain at the curent cell and sensor location is less than ABL height
          if (abs(WGD->domain.z[WGD->terrain_face_id[id]] - WGD->domain.z[WGD->terrain_face_id[site_id[ii]]]) > abl_height[ii]) {
            surf_layer_height = asl_percent * abl_height[ii];
          } else {
            surf_layer_height = asl_percent * (2 * abl_height[ii] - abs(WGD->domain.z[WGD->terrain_face_id[id]] - WGD->domain.z[WGD->terrain_face_id[site_id[ii]]]));
          }
          // If sum of z index and the terrain index at the sensor location is outside the domain
          if (k + WGD->terrain_face_id[site_id[ii]] > nz - 2) {
            sum_wu += wm[ii][i][j] * (u_prof[ii * nz + nz - 2] - u0_int[nz - 2 + ii * nz]);
            sum_wv += wm[ii][i][j] * (v_prof[ii * nz + nz - 2] - v0_int[nz - 2 + ii * nz]);
            sum_wm += wm[ii][i][j];
          }// If height (above ground) is less than or equal to ASL height
          else if ((WGD->domain.z[k_mod]-z_terrain) <= surf_layer_height) {
            sum_wu += wm[ii][i][j]
                      * (u_prof[ii * nz + k + WGD->terrain_face_id[site_id[ii]]]
                         - u0_int[k + WGD->terrain_face_id[site_id[ii]] + ii * nz]);
            sum_wv += wm[ii][i][j]
                      * (v_prof[ii * nz + k + WGD->terrain_face_id[site_id[ii]]]
                         - v0_int[k + WGD->terrain_face_id[site_id[ii]] + ii * nz]);
            sum_wm += wm[ii][i][j];
          }// If height (above ground) is greater than ASL height and modified index is inside the domain
          else if ((WGD->domain.z[k_mod]-z_terrain) > surf_layer_height
                   && k + WGD->terrain_face_id[site_id[ii]] < nz
                   && k_mod > k + WGD->terrain_face_id[site_id[ii]]) {
            sum_wu += wm[ii][i][j] * (u_prof[ii * nz + k_mod] - u0_int[k_mod + ii * nz]);
            sum_wv += wm[ii][i][j] * (v_prof[ii * nz + k_mod] - v0_int[k_mod + ii * nz]);
            sum_wm += wm[ii][i][j];
	    } else {
            sum_wu += wm[ii][i][j]
                      * (u_prof[ii * nz + k + WGD->terrain_face_id[site_id[ii]]]
                         - u0_int[k + WGD->terrain_face_id[site_id[ii]] + ii * nz]);
            sum_wv += wm[ii][i][j]
                      * (v_prof[ii * nz + k + WGD->terrain_face_id[site_id[ii]]]
                         - v0_int[k + WGD->terrain_face_id[site_id[ii]] + ii * nz]);
            sum_wm += wm[ii][i][j];
          }
        }

        if (sum_wm != 0) {
          icell_face = WGD->domain.face(i, j, k_mod);
          WGD->u0[icell_face] = WGD->u0[icell_face] + sum_wu / sum_wm;
          WGD->v0[icell_face] = WGD->v0[icell_face] + sum_wv / sum_wm;
        }
      }
    }
  }
}
