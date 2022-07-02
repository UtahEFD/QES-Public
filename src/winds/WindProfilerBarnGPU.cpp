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
#include "WindProfilerBarnGPU.h"


void WindProfilerBarnGPU::interpolateWindProfile(const WINDSInputData *WID, WINDSGeneralData *WGD)
{
  std::cout << "new Wind Interp" << std::endl;

  const float vk = 0.4;// Von Karman's constant
  float canopy_d, u_H;
  float site_UTM_x, site_UTM_y;
  float site_lon, site_lat;
  float wind_dir, z0_new, z0_high, z0_low;
  float psi, psi_first, x_temp, u_star;
  float u_new, u_new_low, u_new_high;
  int log_flag, iter, id;
  float a1, a2, a3;
  float site_mag;
  float blending_height = 0.0, average__one_overL = 0.0;
  int max_terrain = 1;
  std::vector<float> x, y;

  int num_sites = WID->metParams->sensors.size();
  // array that specifies which timestep of the sensor is related to the running timestep of the code
  // if the value is -1, means that the timestep information is missing for the sensor
  std::vector<int> time_id(num_sites, -1);

  // FM TEMPORARY!!!!!!!!!!!!!
  int solverType = 2;
  time_id = WGD->time_id;

  // loop to adjust the namber of sensors have information for the running timestep of the code
  for (auto i = 0; i < time_id.size(); i++) {
    if (time_id[i] == -1) {
      num_sites -= 1;
    }
  }

  std::vector<int> available_sensor_id;
  for (auto i = 0; i < time_id.size(); i++) {
    if (time_id[i] != -1) {
      available_sensor_id.push_back(i);
    }
  }

  std::vector<std::vector<float>> u_prof(num_sites, std::vector<float>(WGD->nz, 0.0));
  std::vector<std::vector<float>> v_prof(num_sites, std::vector<float>(WGD->nz, 0.0));
  int icell_face, icell_cent;

  std::vector<int> site_i(num_sites, 0);
  std::vector<int> site_j(num_sites, 0);
  std::vector<int> site_id(num_sites, 0);
  std::vector<float> site_theta(num_sites, 0.0);
  int count = 0;

  // Loop through all sites and create velocity profiles (WGD->u0,WGD->v0)
  for (auto i = 0; i < WID->metParams->sensors.size(); i++) {
    // If sensor does not have the timestep information, skip it
    if (time_id[i] == -1) {
      count += 1;
      continue;
    }

    float convergence = 0.0;

    average__one_overL += WID->metParams->sensors[i]->TS[time_id[i]]->site_one_overL / num_sites;
    if (WID->simParams->UTMx != 0 && WID->simParams->UTMy != 0) {
      if (WID->metParams->sensors[i]->site_coord_flag == 1) {
        WID->metParams->sensors[i]->site_UTM_x = WID->metParams->sensors[i]->site_xcoord * acos(WGD->theta) + WID->metParams->sensors[i]->site_ycoord * asin(WGD->theta) + WID->simParams->UTMx;
        WID->metParams->sensors[i]->site_UTM_y = WID->metParams->sensors[i]->site_xcoord * asin(WGD->theta) + WID->metParams->sensors[i]->site_ycoord * acos(WGD->theta) + WID->simParams->UTMy;
        WID->metParams->sensors[i]->site_UTM_zone = WID->simParams->UTMZone;
        // Calling UTMConverter function to convert UTM coordinate to lat/lon and vice versa (located in Sensor.cpp)
        GIStool::UTMConverter(WID->metParams->sensors[i]->site_lon, WID->metParams->sensors[i]->site_lat, WID->metParams->sensors[i]->site_UTM_x, WID->metParams->sensors[i]->site_UTM_y, WID->metParams->sensors[i]->site_UTM_zone, 1);
      }

      if (WID->metParams->sensors[i]->site_coord_flag == 2) {
        // Calling UTMConverter function to convert UTM coordinate to lat/lon and vice versa (located in Sensor.cpp)
        GIStool::UTMConverter(WID->metParams->sensors[i]->site_lon, WID->metParams->sensors[i]->site_lat, WID->metParams->sensors[i]->site_UTM_x, WID->metParams->sensors[i]->site_UTM_y, WID->metParams->sensors[i]->site_UTM_zone, 1);
        WID->metParams->sensors[i]->site_xcoord = WID->metParams->sensors[i]->site_UTM_x - WID->simParams->UTMx;
        WID->metParams->sensors[i]->site_ycoord = WID->metParams->sensors[i]->site_UTM_y - WID->simParams->UTMy;
      }
      GIStool::getConvergence(WID->metParams->sensors[i]->site_lon, WID->metParams->sensors[i]->site_lat, WID->metParams->sensors[i]->site_UTM_zone, convergence);
    }

    int idx = i - count;// id of the available sensors for the running timestep of the code
    site_theta[idx] = (270.0 - WID->metParams->sensors[i]->TS[time_id[i]]->site_wind_dir[0]) * M_PI / 180.0;


    site_i[idx] = WID->metParams->sensors[i]->site_xcoord / WGD->dx;
    site_j[idx] = WID->metParams->sensors[i]->site_ycoord / WGD->dy;
    site_id[idx] = site_i[idx] + site_j[idx] * WGD->nx;
    int id = 1;
    int counter = 0;
    if (WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[0] > 0) {
      blending_height += WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[0] / num_sites;
    } else {
      if (WID->metParams->sensors[i]->TS[time_id[i]]->site_blayer_flag == 4) {
        while (id < WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref.size() && WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[id] > 0 && counter < 1) {
          blending_height += WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[id] / num_sites;
          counter += 1;
          id += 1;
        }
      }
    }

    // If site has a uniform velocity profile
    if (WID->metParams->sensors[i]->TS[time_id[i]]->site_blayer_flag == 0) {
      for (auto k = WGD->terrain_face_id[site_id[idx]]; k < WGD->nz; k++) {
        u_prof[idx][k] = cos(site_theta[idx]) * WID->metParams->sensors[i]->TS[time_id[i]]->site_U_ref[0];
        v_prof[idx][k] = sin(site_theta[idx]) * WID->metParams->sensors[i]->TS[time_id[i]]->site_U_ref[0];
      }
    }
    // Logarithmic velocity profile
    if (WID->metParams->sensors[i]->TS[time_id[i]]->site_blayer_flag == 1) {
      // This loop should be bounded by size of the z
      // vector, and not WGD->nz since z.size can be equal to
      // WGD->nz+1 from what I can tell.  We access z[k]
      // below...
      for (auto k = WGD->terrain_face_id[site_id[idx]]; k < WGD->z.size(); k++) {
        if (k == WGD->terrain_face_id[site_id[idx]]) {
          if (WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[0] * WID->metParams->sensors[i]->TS[time_id[i]]->site_one_overL >= 0) {
            psi = 4.7 * WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[0] * WID->metParams->sensors[i]->TS[time_id[i]]->site_one_overL;
          } else {
            x_temp = pow((1.0 - 15.0 * WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[0] * WID->metParams->sensors[i]->TS[time_id[i]]->site_one_overL), 0.25);
            psi = -2.0 * log(0.5 * (1.0 + x_temp)) - log(0.5 * (1.0 + pow(x_temp, 2.0))) + 2.0 * atan(x_temp) - 0.5 * M_PI;
          }

          u_star = WID->metParams->sensors[i]->TS[time_id[i]]->site_U_ref[0] * vk / (log((WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[0] + WID->metParams->sensors[i]->TS[time_id[i]]->site_z0) / WID->metParams->sensors[i]->TS[time_id[i]]->site_z0) + psi);
        }
        if ((WGD->z[k] - WGD->z_face[WGD->terrain_face_id[site_id[idx]] - 1]) * WID->metParams->sensors[i]->TS[time_id[i]]->site_one_overL >= 0) {
          psi = 4.7 * (WGD->z[k] - WGD->z_face[WGD->terrain_face_id[site_id[idx]] - 1]) * WID->metParams->sensors[i]->TS[time_id[i]]->site_one_overL;
        } else {
          x_temp = pow((1.0 - 15.0 * (WGD->z[k] - WGD->z_face[WGD->terrain_face_id[site_id[idx]] - 1]) * WID->metParams->sensors[i]->TS[time_id[i]]->site_one_overL), 0.25);
          psi = -2.0 * log(0.5 * (1.0 + x_temp)) - log(0.5 * (1.0 + pow(x_temp, 2.0))) + 2.0 * atan(x_temp) - 0.5 * M_PI;
        }

        u_prof[idx][k] = (cos(site_theta[idx]) * u_star / vk) * (log(((WGD->z[k] - WGD->z_face[WGD->terrain_face_id[site_id[idx]] - 1]) + WID->metParams->sensors[i]->TS[time_id[i]]->site_z0) / WID->metParams->sensors[i]->TS[time_id[i]]->site_z0) + psi);
        v_prof[idx][k] = (sin(site_theta[idx]) * u_star / vk) * (log(((WGD->z[k] - WGD->z_face[WGD->terrain_face_id[site_id[idx]] - 1]) + WID->metParams->sensors[i]->TS[time_id[i]]->site_z0) / WID->metParams->sensors[i]->TS[time_id[i]]->site_z0) + psi);
      }
    }

    // Exponential velocity profile
    if (WID->metParams->sensors[i]->TS[time_id[i]]->site_blayer_flag == 2) {
      for (auto k = WGD->terrain_face_id[site_id[idx]]; k < WGD->nz; k++) {
        u_prof[idx][k] = cos(site_theta[idx]) * WID->metParams->sensors[i]->TS[time_id[i]]->site_U_ref[0] * pow(((WGD->z[k] - WGD->z_face[WGD->terrain_face_id[site_id[idx]] - 1]) / WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[0]), WID->metParams->sensors[i]->TS[time_id[i]]->site_p);
        v_prof[idx][k] = sin(site_theta[idx]) * WID->metParams->sensors[i]->TS[time_id[i]]->site_U_ref[0] * pow(((WGD->z[k] - WGD->z_face[WGD->terrain_face_id[site_id[idx]] - 1]) / WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[0]), WID->metParams->sensors[i]->TS[time_id[i]]->site_p);
      }
    }

    // Canopy velocity profile
    if (WID->metParams->sensors[i]->TS[time_id[i]]->site_blayer_flag == 3) {
      for (auto k = WGD->terrain_face_id[site_id[idx]]; k < WGD->nz; k++) {
        if (k == WGD->terrain_face_id[site_id[idx]]) {
          if (WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[0] * WID->metParams->sensors[i]->TS[time_id[i]]->site_one_overL > 0) {
            psi = 4.7 * WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[0] * WID->metParams->sensors[i]->TS[time_id[i]]->site_one_overL;
          } else {
            x_temp = pow((1.0 - 15.0 * WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[0] * WID->metParams->sensors[i]->TS[time_id[i]]->site_one_overL), 0.25);
            psi = -2.0 * log(0.5 * (1.0 + x_temp)) - log(0.5 * (1.0 + pow(x_temp, 2.0))) + 2.0 * atan(x_temp) - 0.5 * M_PI;
          }
          u_star = WID->metParams->sensors[i]->TS[time_id[i]]->site_U_ref[0] * vk / (log(WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[0] / WID->metParams->sensors[i]->TS[time_id[i]]->site_z0) + psi);
          canopy_d = WGD->canopyBisection(u_star, WID->metParams->sensors[i]->TS[time_id[i]]->site_z0, WID->metParams->sensors[i]->TS[time_id[i]]->site_canopy_H, WID->metParams->sensors[i]->TS[time_id[i]]->site_atten_coeff, vk, psi);
          if (WID->metParams->sensors[i]->TS[time_id[i]]->site_canopy_H * WID->metParams->sensors[i]->TS[time_id[i]]->site_one_overL > 0) {
            psi = 4.7 * (WID->metParams->sensors[i]->TS[time_id[i]]->site_canopy_H - canopy_d) * WID->metParams->sensors[i]->TS[time_id[i]]->site_one_overL;
          } else {
            x_temp = pow((1.0 - 15.0 * (WID->metParams->sensors[i]->TS[time_id[i]]->site_canopy_H - canopy_d) * WID->metParams->sensors[i]->TS[time_id[i]]->site_one_overL), 0.25);
            psi = -2.0 * log(0.5 * (1.0 + x_temp)) - log(0.5 * (1.0 + pow(x_temp, 2.0))) + 2.0 * atan(x_temp) - 0.5 * M_PI;
          }
          u_H = (u_star / vk) * (log((WID->metParams->sensors[i]->TS[time_id[i]]->site_canopy_H - canopy_d) / WID->metParams->sensors[i]->TS[time_id[i]]->site_z0) + psi);
          if (WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[0] < WID->metParams->sensors[i]->TS[time_id[i]]->site_canopy_H) {
            WID->metParams->sensors[i]->TS[time_id[i]]->site_U_ref[0] /= u_H * exp(WID->metParams->sensors[i]->TS[time_id[i]]->site_atten_coeff * (WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[0] / WID->metParams->sensors[i]->TS[time_id[i]]->site_canopy_H) - 1.0);
          } else {
            if (WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[0] * WID->metParams->sensors[i]->TS[time_id[i]]->site_one_overL > 0) {
              psi = 4.7 * (WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[0] - canopy_d) * WID->metParams->sensors[i]->TS[time_id[i]]->site_one_overL;
            } else {
              x_temp = pow(1.0 - 15.0 * (WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[0] - canopy_d) * WID->metParams->sensors[i]->TS[time_id[i]]->site_one_overL, 0.25);
              psi = -2.0 * log(0.5 * (1.0 + x_temp)) - log(0.5 * (1.0 + pow(x_temp, 2.0))) + 2.0 * atan(x_temp) - 0.5 * M_PI;
            }
            WID->metParams->sensors[i]->TS[time_id[i]]->site_U_ref[0] /= ((u_star / vk) * (log((WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[0] - canopy_d) / WID->metParams->sensors[i]->TS[time_id[i]]->site_z0) + psi));
          }
          u_star *= WID->metParams->sensors[i]->TS[time_id[i]]->site_U_ref[0];
          u_H *= WID->metParams->sensors[i]->TS[time_id[i]]->site_U_ref[0];
        }

        if ((WGD->z[k] - WGD->z_face[WGD->terrain_face_id[site_id[idx]] - 1]) < WID->metParams->sensors[i]->TS[time_id[i]]->site_canopy_H) {
          u_prof[idx][k] = cos(site_theta[idx]) * u_H * exp(WID->metParams->sensors[i]->TS[time_id[i]]->site_atten_coeff * (((WGD->z[k] - WGD->z_face[WGD->terrain_face_id[site_id[idx]] - 1]) / WID->metParams->sensors[i]->TS[time_id[i]]->site_canopy_H) - 1.0));
          v_prof[idx][k] = sin(site_theta[idx]) * u_H * exp(WID->metParams->sensors[i]->TS[time_id[i]]->site_atten_coeff * (((WGD->z[k] - WGD->z_face[WGD->terrain_face_id[site_id[idx]] - 1]) / WID->metParams->sensors[i]->TS[time_id[i]]->site_canopy_H) - 1.0));
        }
        if ((WGD->z[k] - WGD->z_face[WGD->terrain_face_id[site_id[idx]] - 1]) > WID->metParams->sensors[i]->TS[time_id[i]]->site_canopy_H) {
          if ((WGD->z[k] - WGD->z_face[WGD->terrain_face_id[site_id[idx]] - 1]) * WID->metParams->sensors[i]->TS[time_id[i]]->site_one_overL > 0) {
            psi = 4.7 * ((WGD->z[k] - WGD->z_face[WGD->terrain_face_id[site_id[idx]] - 1]) - canopy_d) * WID->metParams->sensors[i]->TS[time_id[i]]->site_one_overL;
          } else {
            x_temp = pow(1.0 - 15.0 * ((WGD->z[k] - WGD->z_face[WGD->terrain_face_id[site_id[idx]] - 1]) - canopy_d) * WID->metParams->sensors[i]->TS[time_id[i]]->site_one_overL, 0.25);
            psi = -2.0 * log(0.5 * (1.0 + x_temp)) - log(0.5 * (1.0 + pow(x_temp, 2.0))) + 2.0 * atan(x_temp) - 0.5 * M_PI;
          }
          u_prof[idx][k] = (cos(site_theta[idx]) * u_star / vk) * (log(((WGD->z[k] - WGD->z_face[WGD->terrain_face_id[site_id[idx]] - 1]) - canopy_d) / WID->metParams->sensors[i]->TS[time_id[i]]->site_z0) + psi);
          v_prof[idx][k] = (sin(site_theta[idx]) * u_star / vk) * (log(((WGD->z[k] - WGD->z_face[WGD->terrain_face_id[site_id[idx]] - 1]) - canopy_d) / WID->metParams->sensors[i]->TS[time_id[i]]->site_z0) + psi);
        }
      }
    }

    // Data entry profile (WRF output)
    if (WID->metParams->sensors[i]->TS[time_id[i]]->site_blayer_flag == 4) {
      int z_size = WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref.size();
      int ii = -1;
      site_theta[idx] = (270.0 - WID->metParams->sensors[i]->TS[time_id[i]]->site_wind_dir[0]) * M_PI / 180.0;

      // Needs to be nz-1 for [0, n-1] indexing
      for (auto k = WGD->terrain_face_id[site_id[idx]]; k < WGD->nz - 1; k++) {
        if ((WGD->z[k] - WGD->z_face[WGD->terrain_face_id[site_id[idx]] - 1]) < WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[0] || z_size == 1) {
          u_prof[idx][k] = (WID->metParams->sensors[i]->TS[time_id[i]]->site_U_ref[0] * cos(site_theta[idx]) / log((WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[0] + WID->metParams->sensors[i]->TS[time_id[i]]->site_z0) / WID->metParams->sensors[i]->TS[time_id[i]]->site_z0))
                           * log(((WGD->z[k] - WGD->z_face[WGD->terrain_face_id[site_id[idx]] - 1]) + WID->metParams->sensors[i]->TS[time_id[i]]->site_z0) / WID->metParams->sensors[i]->TS[time_id[i]]->site_z0);
          v_prof[idx][k] = (WID->metParams->sensors[i]->TS[time_id[i]]->site_U_ref[0] * sin(site_theta[idx]) / log((WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[0] + WID->metParams->sensors[i]->TS[time_id[i]]->site_z0) / WID->metParams->sensors[i]->TS[time_id[i]]->site_z0))
                           * log(((WGD->z[k] - WGD->z_face[WGD->terrain_face_id[site_id[idx]] - 1]) + WID->metParams->sensors[i]->TS[time_id[i]]->site_z0) / WID->metParams->sensors[i]->TS[time_id[i]]->site_z0);
        } else {

          if ((ii < z_size - 2) && ((WGD->z[k] - WGD->z_face[WGD->terrain_face_id[site_id[idx]] - 1]) >= WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii + 1])) {
            ii += 1;
            if (abs(WID->metParams->sensors[i]->TS[time_id[i]]->site_wind_dir[ii + 1] - WID->metParams->sensors[i]->TS[time_id[i]]->site_wind_dir[ii]) > 180.0) {
              if (WID->metParams->sensors[i]->TS[time_id[i]]->site_wind_dir[ii + 1] > WID->metParams->sensors[i]->TS[time_id[i]]->site_wind_dir[ii]) {
                wind_dir = (WID->metParams->sensors[i]->TS[time_id[i]]->site_wind_dir[ii + 1] - 360.0 - WID->metParams->sensors[i]->TS[time_id[i]]->site_wind_dir[ii + 1])
                           / (WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii + 1] - WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii]);
              } else {
                wind_dir = (WID->metParams->sensors[i]->TS[time_id[i]]->site_wind_dir[ii + 1] + 360.0 - WID->metParams->sensors[i]->TS[time_id[i]]->site_wind_dir[ii + 1])
                           / (WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii + 1] - WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii]);
              }
            } else {
              wind_dir = (WID->metParams->sensors[i]->TS[time_id[i]]->site_wind_dir[ii + 1] - WID->metParams->sensors[i]->TS[time_id[i]]->site_wind_dir[ii])
                         / (WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii + 1] - WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii]);
            }
            z0_high = 20.0;
            u_star = vk * WID->metParams->sensors[i]->TS[time_id[i]]->site_U_ref[ii] / log((WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii] + z0_high) / z0_high);
            u_new_high = (u_star / vk) * log((WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii] + z0_high) / z0_high);
            z0_low = 1e-9;
            u_star = vk * WID->metParams->sensors[i]->TS[time_id[i]]->site_U_ref[ii] / log((WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii] + z0_low) / z0_low);
            u_new_low = (u_star / vk) * log((WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii + 1] + z0_low) / z0_low);

            if (WID->metParams->sensors[i]->TS[time_id[i]]->site_U_ref[ii + 1] > u_new_low && WID->metParams->sensors[i]->TS[time_id[i]]->site_U_ref[ii + 1] < u_new_high) {
              log_flag = 1;
              iter = 0;
              u_star = vk * WID->metParams->sensors[i]->TS[time_id[i]]->site_U_ref[ii] / log((WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii] + WID->metParams->sensors[i]->TS[time_id[i]]->site_z0) / WID->metParams->sensors[i]->TS[time_id[i]]->site_z0);
              u_new = (u_star / vk) * log((WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii + 1] + WID->metParams->sensors[i]->TS[time_id[i]]->site_z0) / WID->metParams->sensors[i]->TS[time_id[i]]->site_z0);
              while (iter < 200 && abs(u_new - WID->metParams->sensors[i]->TS[time_id[i]]->site_U_ref[ii]) > 0.0001 * WID->metParams->sensors[i]->TS[time_id[i]]->site_U_ref[ii]) {
                iter += 1;
                z0_new = 0.5 * (z0_low + z0_high);
                u_star = vk * WID->metParams->sensors[i]->TS[time_id[i]]->site_U_ref[ii] / log((WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii] + z0_new) / z0_new);
                u_new = (u_star / vk) * log((WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii + 1] + z0_new) / z0_new);
                if (u_new > WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii + 1]) {
                  z0_high = z0_new;
                } else {
                  z0_low = z0_new;
                }
              }
            } else {
              log_flag = 0;
              if (ii < z_size - 2) {
                a1 = ((WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii + 1] - WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii])
                        * (WID->metParams->sensors[i]->TS[time_id[i]]->site_U_ref[ii + 2] - WID->metParams->sensors[i]->TS[time_id[i]]->site_U_ref[ii])
                      + (WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii] - WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii + 2])
                          * (WID->metParams->sensors[i]->TS[time_id[i]]->site_U_ref[ii + 1] - WID->metParams->sensors[i]->TS[time_id[i]]->site_U_ref[ii]))
                     / ((WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii + 1] - WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii])
                          * (pow(WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii + 2], 2.0) - pow(WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii], 2.0))
                        + (pow(WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii + 1], 2.0) - pow(WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii], 2.0))
                            * (WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii] - WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii + 2]));
              } else {
                a1 = 0.0;
              }
              a2 = ((WID->metParams->sensors[i]->TS[time_id[i]]->site_U_ref[ii + 1] - WID->metParams->sensors[i]->TS[time_id[i]]->site_U_ref[ii])
                    - a1 * (pow(WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii + 1], 2.0) - pow(WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii], 2.0)))
                   / (WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii + 1] - WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii]);
              a3 = WID->metParams->sensors[i]->TS[time_id[i]]->site_U_ref[ii] - a1 * pow(WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii], 2.0)
                   - a2 * WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii];
            }
          }
          if (log_flag == 1) {
            site_mag = (u_star / vk) * log(((WGD->z[k] - WGD->z_face[WGD->terrain_face_id[site_id[idx]] - 1]) + z0_new) / z0_new);
          } else {
            site_mag = a1 * pow((WGD->z[k] - WGD->z_face[WGD->terrain_face_id[site_id[idx]] - 1]), 2.0) + a2 * (WGD->z[k] - WGD->z_face[WGD->terrain_face_id[site_id[idx]] - 1]) + a3;
          }
          site_theta[idx] = (270.0 - (WID->metParams->sensors[i]->TS[time_id[i]]->site_wind_dir[ii] + wind_dir * ((WGD->z[k] - WGD->z_face[WGD->terrain_face_id[site_id[idx]] - 1]) - WID->metParams->sensors[i]->TS[time_id[i]]->site_z_ref[ii]))) * M_PI / 180.0;
          u_prof[idx][k] = site_mag * cos(site_theta[idx]);
          v_prof[idx][k] = site_mag * sin(site_theta[idx]);
        }
      }
    }
  }

  x.resize(WGD->nx);
  for (size_t i = 0; i < WGD->nx; i++) {
    x[i] = (i - 0.5) * WGD->dx; /**< Location of face centers in x-dir */
  }

  y.resize(WGD->ny);
  for (auto j = 0; j < WGD->ny; j++) {
    y[j] = (j - 0.5) * WGD->dy; /**< Location of face centers in y-dir */
  }

  int k_mod;
  if (num_sites == 1) {
    for (auto k = 0; k < WGD->nz; k++) {
      for (auto j = 0; j < WGD->ny; j++) {
        for (auto i = 0; i < WGD->nx; i++) {

          int id = i + j * WGD->nx;
          if (k + WGD->terrain_face_id[id] - 1 < WGD->nz) {
            k_mod = k + WGD->terrain_face_id[id] - 1;
          } else {
            continue;
          }
          icell_face = i + j * WGD->nx + k_mod * WGD->nx * WGD->ny;/// Lineralized index for cell faced values
          if (k + WGD->terrain_face_id[site_id[0]] - 1 > WGD->nz - 2) {
            WGD->u0[icell_face] = u_prof[0][WGD->nz - 2];
            WGD->v0[icell_face] = v_prof[0][WGD->nz - 2];
          } else {
            WGD->u0[icell_face] = u_prof[0][k + WGD->terrain_face_id[site_id[0]] - 1];
            WGD->v0[icell_face] = v_prof[0][k + WGD->terrain_face_id[site_id[0]] - 1];
          }

          // WGD->w0[icell_face] = 0.0;         /// Perpendicular wind direction
        }
      }
    }
  }

  // If number of sites are more than one
  // Apply 2D Barnes scheme to interpolate site velocity profiles to the whole domain
  //
  // If number of sites are more than one
  // Apply 2D Barnes scheme to interpolate site velocity profiles to the whole domain

  auto startBarnesGPU = std::chrono::high_resolution_clock::now();
  BarnesInterpolationGPU(WID, WGD, u_prof, v_prof, site_id, num_sites, available_sensor_id);
  auto finishBarnesGPU = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> elapsedBarnesGPU = finishBarnesGPU - startBarnesGPU;
  std::cout << "Elapsed time for Barnes interpolation on GPU: " << elapsedBarnesGPU.count() << " s\n";

  return;
}
