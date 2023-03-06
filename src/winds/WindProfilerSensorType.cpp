/****************************************************************************
 * Copyright (c) 2022 University of Utah
 * Copyright (c) 2022 University of Minnesota Duluth
 *
 * Copyright (c) 2022 Behnam Bozorgmehr
 * Copyright (c) 2022 Jeremy A. Gibbs
 * Copyright (c) 2022 Fabien Margairaz
 * Copyright (c) 2022 Eric R. Pardyjak
 * Copyright (c) 2022 Zachary Patterson
 * Copyright (c) 2022 Rob Stoll
 * Copyright (c) 2022 Lucas Ulmer
 * Copyright (c) 2022 Pete Willemsen
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


void WindProfilerSensorType::sensorsProfiles(const WINDSInputData *WID, WINDSGeneralData *WGD)
{
  const float vk = 0.4;// Von Karman's constant
  float canopy_d = 0.0, u_H = 0.0;
  //float site_UTM_x, site_UTM_y;
  //float site_lon, site_lat;
  float wind_dir = 0.0, z0_new = 0.0, z0_high = 0.0, z0_low = 0.0;
  float psi;// unused: psi_first
  float x_temp;
  float u_star = 0.0, u_new = 0.0, u_new_low = 0.0, u_new_high = 0.0;
  int log_flag = 0, iter = 0;
  float a1 = 0.0, a2 = 0.0, a3 = 0.0;
  float site_mag;
  float blending_height = 0.0, average__one_overL = 0.0;
  // unused: int max_terrain = 1;
  std::vector<float> x, y;

  int num_sites = WID->metParams->sensors.size();
  // array that specifies which timestep of the sensor is related to the running timestep of the code
  // if the value is -1, means that the timestep information is missing for the sensor
  std::vector<int> time_id(num_sites, -1);

  // FM TEMPORARY!!!!!!!!!!!!!
  time_id = WGD->time_id;

  // loop to adjust the namber of sensors have information for the running timestep of the code
  for (auto i = 0u; i < time_id.size(); i++) {
    if (time_id[i] == -1) {
      num_sites -= 1;
    }
  }

  //std::vector<int> available_sensor_id;
  available_sensor_id.clear();
  for (auto i = 0u; i < time_id.size(); i++) {
    if (time_id[i] != -1) {
      available_sensor_id.push_back(i);
    }
  }

  //std::vector<std::vector<float>> u_prof(num_sites, std::vector<float>(WGD->nz, 0.0));
  //std::vector<std::vector<float>> v_prof(num_sites, std::vector<float>(WGD->nz, 0.0));
  //std::vector<float> u_prof(num_sites * WGD->nz, 0.0);
  //std::vector<float> v_prof(num_sites * WGD->nz, 0.0);
  u_prof.clear();
  u_prof.resize(num_sites * WGD->nz, 0.0);
  v_prof.clear();
  v_prof.resize(num_sites * WGD->nz, 0.0);

  site_id.clear();
  site_id.resize(num_sites, 0);

  std::vector<int> site_i(num_sites, 0);
  std::vector<int> site_j(num_sites, 0);
  std::vector<float> site_theta(num_sites, 0.0);
  int count = 0;

  // Loop through all sites and create velocity profiles (WGD->u0,WGD->v0)
  for (auto i = 0u; i < WID->metParams->sensors.size(); i++) {
    // If sensor does not have the timestep information, skip it
    if (time_id[i] == -1) {
      count += 1;
      continue;
    }

    float convergence = 0.0;

    TimeSeries *ts = WID->metParams->sensors[i]->TS[time_id[i]];

    average__one_overL += ts->site_one_overL / num_sites;

    // THIS SOULD NOT BE HERE!!! AND ITS NOT FINISHED
    if (WID->simParams->UTMx != 0 && WID->simParams->UTMy != 0) {
      if (WID->metParams->sensors[i]->site_coord_flag == 1) {
        WID->metParams->sensors[i]->site_UTM_x = WID->metParams->sensors[i]->site_xcoord * acos(WGD->theta)
                                                 + WID->metParams->sensors[i]->site_ycoord * asin(WGD->theta)
                                                 + WID->simParams->UTMx;
        WID->metParams->sensors[i]->site_UTM_y = WID->metParams->sensors[i]->site_xcoord * asin(WGD->theta)
                                                 + WID->metParams->sensors[i]->site_ycoord * acos(WGD->theta)
                                                 + WID->simParams->UTMy;
        WID->metParams->sensors[i]->site_UTM_zone = WID->simParams->UTMZone;
        // Calling UTMConverter function to convert UTM coordinate to lat/lon and vice versa
        GIStool::UTMConverter(WID->metParams->sensors[i]->site_lon,
                              WID->metParams->sensors[i]->site_lat,
                              WID->metParams->sensors[i]->site_UTM_x,
                              WID->metParams->sensors[i]->site_UTM_y,
                              WID->metParams->sensors[i]->site_UTM_zone,
                              true,
                              1);
      }

      if (WID->metParams->sensors[i]->site_coord_flag == 2) {
        // Calling UTMConverter function to convert UTM coordinate to lat/lon and vice versa
        GIStool::UTMConverter(WID->metParams->sensors[i]->site_lon,
                              WID->metParams->sensors[i]->site_lat,
                              WID->metParams->sensors[i]->site_UTM_x,
                              WID->metParams->sensors[i]->site_UTM_y,
                              WID->metParams->sensors[i]->site_UTM_zone,
                              true,
                              1);
        WID->metParams->sensors[i]->site_xcoord = WID->metParams->sensors[i]->site_UTM_x - WID->simParams->UTMx;
        WID->metParams->sensors[i]->site_ycoord = WID->metParams->sensors[i]->site_UTM_y - WID->simParams->UTMy;
      }
      GIStool::getConvergence(WID->metParams->sensors[i]->site_lon,
                              WID->metParams->sensors[i]->site_lat,
                              WID->metParams->sensors[i]->site_UTM_zone,
                              convergence);
    }

    int idx = i - count;// id of the available sensors for the running timestep of the code

    site_theta[idx] = (270.0 - ts->site_wind_dir[0]) * M_PI / 180.0;

    site_i[idx] = WID->metParams->sensors[i]->site_xcoord / WGD->dx;
    site_j[idx] = WID->metParams->sensors[i]->site_ycoord / WGD->dy;
    site_id[idx] = site_i[idx] + site_j[idx] * WGD->nx;

    float z_terrain = WGD->z_face[WGD->terrain_face_id[site_id[idx]]];

    size_t id = 1;
    int counter = 0;
    if (ts->site_z_ref[0] > 0) {
      blending_height += ts->site_z_ref[0] / num_sites;
    } else {
      if (ts->site_blayer_flag == 4) {
        while (id < ts->site_z_ref.size() && ts->site_z_ref[id] > 0 && counter < 1) {
          blending_height += ts->site_z_ref[id] / num_sites;
          counter += 1;
          id += 1;
        }
      }
    }

    // If site has a uniform velocity profile
    if (ts->site_blayer_flag == 0) {
      for (auto k = WGD->terrain_face_id[site_id[idx]]; k < WGD->nz; ++k) {
        u_prof[idx * WGD->nz + k] = cos(site_theta[idx]) * ts->site_U_ref[0];
        v_prof[idx * WGD->nz + k] = sin(site_theta[idx]) * ts->site_U_ref[0];
      }
    }
    // Logarithmic velocity profile
    if (ts->site_blayer_flag == 1) {
      // This loop should be bounded by size of the z
      // vector, and not WGD->nz since z.size can be equal to
      // WGD->nz+1 from what I can tell.  We access z[k]
      // below...
      for (auto k = WGD->terrain_face_id[site_id[idx]]; k < WGD->nz - 1; ++k) {
        if (k == WGD->terrain_face_id[site_id[idx]]) {
          if (ts->site_z_ref[0] * ts->site_one_overL >= 0) {
            psi = 4.7 * ts->site_z_ref[0] * ts->site_one_overL;
          } else {
            x_temp = pow((1.0 - 15.0 * ts->site_z_ref[0] * ts->site_one_overL), 0.25);
            psi = -2.0 * log(0.5 * (1.0 + x_temp)) - log(0.5 * (1.0 + pow(x_temp, 2.0))) + 2.0 * atan(x_temp) - 0.5 * M_PI;
          }

          u_star = ts->site_U_ref[0] * vk / (log((ts->site_z_ref[0] + ts->site_z0) / ts->site_z0) + psi);
        }
        if ((WGD->z[k] - z_terrain) * ts->site_one_overL >= 0) {
          psi = 4.7 * (WGD->z[k] - z_terrain) * ts->site_one_overL;
        } else {
          x_temp = pow((1.0 - 15.0 * (WGD->z[k] - z_terrain) * ts->site_one_overL), 0.25);
          psi = -2.0 * log(0.5 * (1.0 + x_temp)) - log(0.5 * (1.0 + pow(x_temp, 2.0))) + 2.0 * atan(x_temp) - 0.5 * M_PI;
        }

        u_prof[idx * WGD->nz + k] = (cos(site_theta[idx]) * u_star / vk) * (log(((WGD->z[k] - z_terrain) + ts->site_z0) / ts->site_z0) + psi);
        v_prof[idx * WGD->nz + k] = (sin(site_theta[idx]) * u_star / vk) * (log(((WGD->z[k] - z_terrain) + ts->site_z0) / ts->site_z0) + psi);
      }
    }

    // Exponential velocity profile
    if (ts->site_blayer_flag == 2) {
      for (auto k = WGD->terrain_face_id[site_id[idx]]; k < WGD->nz - 1; k++) {
        u_prof[idx * WGD->nz + k] = cos(site_theta[idx]) * ts->site_U_ref[0] * pow(((WGD->z[k] - z_terrain) / ts->site_z_ref[0]), ts->site_z0);
        v_prof[idx * WGD->nz + k] = sin(site_theta[idx]) * ts->site_U_ref[0] * pow(((WGD->z[k] - z_terrain) / ts->site_z_ref[0]), ts->site_z0);
      }
    }

    // Canopy velocity profile
    if (ts->site_blayer_flag == 3) {
      for (auto k = WGD->terrain_face_id[site_id[idx]]; k < WGD->nz - 1; k++) {
        if (k == WGD->terrain_face_id[site_id[idx]]) {
          if (ts->site_z_ref[0] * ts->site_one_overL > 0) {
            psi = 4.7 * ts->site_z_ref[0] * ts->site_one_overL;
          } else {
            x_temp = pow((1.0 - 15.0 * ts->site_z_ref[0] * ts->site_one_overL), 0.25);
            psi = -2.0 * log(0.5 * (1.0 + x_temp)) - log(0.5 * (1.0 + pow(x_temp, 2.0))) + 2.0 * atan(x_temp) - 0.5 * M_PI;
          }
          u_star = ts->site_U_ref[0] * vk / (log(ts->site_z_ref[0] / ts->site_z0) + psi);
          canopy_d = WGD->canopyBisection(u_star, ts->site_z0, ts->site_canopy_H, ts->site_atten_coeff, vk, psi);
          if (ts->site_canopy_H * ts->site_one_overL > 0) {
            psi = 4.7 * (ts->site_canopy_H - canopy_d) * ts->site_one_overL;
          } else {
            x_temp = pow((1.0 - 15.0 * (ts->site_canopy_H - canopy_d) * ts->site_one_overL), 0.25);
            psi = -2.0 * log(0.5 * (1.0 + x_temp)) - log(0.5 * (1.0 + pow(x_temp, 2.0))) + 2.0 * atan(x_temp) - 0.5 * M_PI;
          }
          u_H = (u_star / vk) * (log((ts->site_canopy_H - canopy_d) / ts->site_z0) + psi);
          if (ts->site_z_ref[0] < ts->site_canopy_H) {
            ts->site_U_ref[0] /= u_H * exp(ts->site_atten_coeff * (ts->site_z_ref[0] / ts->site_canopy_H) - 1.0);
          } else {
            if (ts->site_z_ref[0] * ts->site_one_overL > 0) {
              psi = 4.7 * (ts->site_z_ref[0] - canopy_d) * ts->site_one_overL;
            } else {
              x_temp = pow(1.0 - 15.0 * (ts->site_z_ref[0] - canopy_d) * ts->site_one_overL, 0.25);
              psi = -2.0 * log(0.5 * (1.0 + x_temp)) - log(0.5 * (1.0 + pow(x_temp, 2.0))) + 2.0 * atan(x_temp) - 0.5 * M_PI;
            }
            ts->site_U_ref[0] /= ((u_star / vk) * (log((ts->site_z_ref[0] - canopy_d) / ts->site_z0) + psi));
          }
          u_star *= ts->site_U_ref[0];
          u_H *= ts->site_U_ref[0];
        }

        if ((WGD->z[k] - z_terrain) < ts->site_canopy_H) {
          u_prof[idx * WGD->nz + k] = cos(site_theta[idx]) * u_H
                                      * exp(ts->site_atten_coeff * (((WGD->z[k] - z_terrain) / ts->site_canopy_H) - 1.0));
          v_prof[idx * WGD->nz + k] = sin(site_theta[idx]) * u_H
                                      * exp(ts->site_atten_coeff * (((WGD->z[k] - z_terrain) / ts->site_canopy_H) - 1.0));
        }
        if ((WGD->z[k] - z_terrain) > ts->site_canopy_H) {
          if ((WGD->z[k] - z_terrain) * ts->site_one_overL > 0) {
            psi = 4.7 * ((WGD->z[k] - z_terrain) - canopy_d) * ts->site_one_overL;
          } else {
            x_temp = pow(1.0 - 15.0 * ((WGD->z[k] - z_terrain) - canopy_d) * ts->site_one_overL, 0.25);
            psi = -2.0 * log(0.5 * (1.0 + x_temp)) - log(0.5 * (1.0 + pow(x_temp, 2.0))) + 2.0 * atan(x_temp) - 0.5 * M_PI;
          }
          u_prof[idx * WGD->nz + k] = (cos(site_theta[idx]) * u_star / vk)
                                      * (log(((WGD->z[k] - z_terrain) - canopy_d) / ts->site_z0) + psi);
          v_prof[idx * WGD->nz + k] = (sin(site_theta[idx]) * u_star / vk)
                                      * (log(((WGD->z[k] - z_terrain) - canopy_d) / ts->site_z0) + psi);
        }
      }
    }

    // Data entry profile (WRF output)
    if (ts->site_blayer_flag == 4) {
      int z_size = ts->site_z_ref.size();
      int ii = -1;
      site_theta[idx] = (270.0 - ts->site_wind_dir[0]) * M_PI / 180.0;

      // Needs to be nz-1 for [0, n-1] indexing
      for (auto k = WGD->terrain_face_id[site_id[idx]]; k < WGD->nz - 1; k++) {
        if ((WGD->z[k] - z_terrain) < ts->site_z_ref[0] || z_size == 1) {
          u_prof[idx * WGD->nz + k] = (ts->site_U_ref[0] * cos(site_theta[idx]) / log((ts->site_z_ref[0] + ts->site_z0) / ts->site_z0))
                                      * log(((WGD->z[k] - z_terrain) + ts->site_z0) / ts->site_z0);
          v_prof[idx * WGD->nz + k] = (ts->site_U_ref[0] * sin(site_theta[idx]) / log((ts->site_z_ref[0] + ts->site_z0) / ts->site_z0))
                                      * log(((WGD->z[k] - z_terrain) + ts->site_z0) / ts->site_z0);
        } else {

          if ((ii < z_size - 2) && ((WGD->z[k] - z_terrain) >= ts->site_z_ref[ii + 1])) {
            ii += 1;
            if (abs(ts->site_wind_dir[ii + 1] - ts->site_wind_dir[ii]) > 180.0) {
              if (ts->site_wind_dir[ii + 1] > ts->site_wind_dir[ii]) {
                wind_dir = (ts->site_wind_dir[ii + 1] - 360.0 - ts->site_wind_dir[ii + 1])
                           / (ts->site_z_ref[ii + 1] - ts->site_z_ref[ii]);
              } else {
                wind_dir = (ts->site_wind_dir[ii + 1] + 360.0 - ts->site_wind_dir[ii + 1])
                           / (ts->site_z_ref[ii + 1] - ts->site_z_ref[ii]);
              }
            } else {
              wind_dir = (ts->site_wind_dir[ii + 1] - ts->site_wind_dir[ii])
                         / (ts->site_z_ref[ii + 1] - ts->site_z_ref[ii]);
            }
            z0_high = 20.0;
            u_star = vk * ts->site_U_ref[ii] / log((ts->site_z_ref[ii] + z0_high) / z0_high);
            u_new_high = (u_star / vk) * log((ts->site_z_ref[ii] + z0_high) / z0_high);
            z0_low = 1e-9;
            u_star = vk * ts->site_U_ref[ii] / log((ts->site_z_ref[ii] + z0_low) / z0_low);
            u_new_low = (u_star / vk) * log((ts->site_z_ref[ii + 1] + z0_low) / z0_low);

            if (ts->site_U_ref[ii + 1] > u_new_low && ts->site_U_ref[ii + 1] < u_new_high) {
              log_flag = 1;
              iter = 0;
              u_star = vk * ts->site_U_ref[ii] / log((ts->site_z_ref[ii] + ts->site_z0) / ts->site_z0);
              u_new = (u_star / vk) * log((ts->site_z_ref[ii + 1] + ts->site_z0) / ts->site_z0);
              while (iter < 200 && abs(u_new - ts->site_U_ref[ii]) > 0.0001 * ts->site_U_ref[ii]) {
                iter += 1;
                z0_new = 0.5 * (z0_low + z0_high);
                u_star = vk * ts->site_U_ref[ii] / log((ts->site_z_ref[ii] + z0_new) / z0_new);
                u_new = (u_star / vk) * log((ts->site_z_ref[ii + 1] + z0_new) / z0_new);
                if (u_new > ts->site_z_ref[ii + 1]) {
                  z0_high = z0_new;
                } else {
                  z0_low = z0_new;
                }
              }
            } else {
              log_flag = 0;
              if (ii < z_size - 2) {
                a1 = ((ts->site_z_ref[ii + 1] - ts->site_z_ref[ii]) * (ts->site_U_ref[ii + 2] - ts->site_U_ref[ii])
                      + (ts->site_z_ref[ii] - ts->site_z_ref[ii + 2]) * (ts->site_U_ref[ii + 1] - ts->site_U_ref[ii]))
                     / ((ts->site_z_ref[ii + 1] - ts->site_z_ref[ii]) * (pow(ts->site_z_ref[ii + 2], 2.0) - pow(ts->site_z_ref[ii], 2.0))
                        + (pow(ts->site_z_ref[ii + 1], 2.0) - pow(ts->site_z_ref[ii], 2.0)) * (ts->site_z_ref[ii] - ts->site_z_ref[ii + 2]));
              } else {
                a1 = 0.0;
              }
              a2 = ((ts->site_U_ref[ii + 1] - ts->site_U_ref[ii]) - a1 * (pow(ts->site_z_ref[ii + 1], 2.0) - pow(ts->site_z_ref[ii], 2.0)))
                   / (ts->site_z_ref[ii + 1] - ts->site_z_ref[ii]);
              a3 = ts->site_U_ref[ii] - a1 * pow(ts->site_z_ref[ii], 2.0) - a2 * ts->site_z_ref[ii];
            }
          }
          if (log_flag == 1) {
            site_mag = (u_star / vk) * log(((WGD->z[k] - z_terrain) + z0_new) / z0_new);
          } else {
            site_mag = a1 * pow((WGD->z[k] - z_terrain), 2.0)
                       + a2 * (WGD->z[k] - z_terrain) + a3;
          }
          site_theta[idx] = (270.0 - (ts->site_wind_dir[ii] + wind_dir * ((WGD->z[k] - z_terrain) - ts->site_z_ref[ii]))) * M_PI / 180.0;
          u_prof[idx * WGD->nz + k] = site_mag * cos(site_theta[idx]);
          v_prof[idx * WGD->nz + k] = site_mag * sin(site_theta[idx]);
        }
      }
    }
  }

  if (average__one_overL > 0.0) {
    // Stable boundary layer
    abl_height = 200;
  } else if (average__one_overL < 0.0) {
    // Unstable boundary layer
    abl_height = 1000;
  } else {
    // Neutral boundary layer
    abl_height = 100;
  }

  return;
}


void WindProfilerSensorType::singleSensorInterpolation(WINDSGeneralData *WGD)

{
  float surf_layer_height;// Surface layer height of the atmospheric boundary layer (ABL)

  for (auto k = 1; k < WGD->nz - 1; ++k) {
    //Set the modified k-index (sensor)
    int k_mod_sens = k + WGD->terrain_face_id[site_id[0]] - 1;
    for (auto j = 0; j < WGD->ny; ++j) {
      for (auto i = 0; i < WGD->nx; ++i) {

        int id = i + j * WGD->nx;//Index in horizontal surface
        int k_mod(0);
        //If height added to top of terrain is still inside QES domain
        if (k + WGD->terrain_face_id[id] - 1 < WGD->nz) {
          //Set the modified k-index (current location)
          k_mod = k + WGD->terrain_face_id[id] - 1;
        } else {
          continue;
        }
        // Lineralized index for cell faced values
        int icell_face = i + j * WGD->nx + k_mod * WGD->nx * WGD->ny;
        // If the height difference between the terrain at the curent cell and sensor location is less than ABL height
        if (abs(WGD->z[WGD->terrain_face_id[id]] - WGD->z[WGD->terrain_face_id[site_id[0]]]) > abl_height) {
          surf_layer_height = asl_percent * abl_height;
        } else {
          surf_layer_height = asl_percent * (2 * abl_height - (WGD->z[WGD->terrain_face_id[id]] - WGD->z[WGD->terrain_face_id[site_id[0]]]));
        }
        // If height (above ground) is less than or equal to ASL height
        if (WGD->z[k] <= surf_layer_height) {
          WGD->u0[icell_face] = u_prof[k_mod_sens];
          WGD->v0[icell_face] = v_prof[k_mod_sens];
        }// If sum of z index and the terrain index at the sensor location is outside the domain
        else if (k + WGD->terrain_face_id[site_id[0]] - 1 > WGD->nz - 2) {
          WGD->u0[icell_face] = u_prof[WGD->nz - 2];
          WGD->v0[icell_face] = v_prof[WGD->nz - 2];
        }// If height (above ground) is greater than ASL height and modified index is inside the domain
        else if (WGD->z[k] > surf_layer_height && k_mod_sens < WGD->nz) {
          WGD->u0[icell_face] = u_prof[k_mod_sens];
          WGD->v0[icell_face] = v_prof[k_mod_sens];
        }

        WGD->w0[icell_face] = 0.0;// Perpendicular wind direction
      }
    }
  }

  return;
}
