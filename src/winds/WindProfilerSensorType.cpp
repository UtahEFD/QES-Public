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
#include "WindProfilerBarnGPU.h"
#include "WindProfilerHRRR.h"


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
  float blending_height = 0.0, average_one_overL = 0.0;
  // unused: int max_terrain = 1;
  std::vector<float> x, y;
  float z_terrain;

  int num_sites = WID->metParams->sensors.size();
  // array that specifies which timestep of the sensor is related to the running timestep of the code
  // if the value is -1, means that the timestep information is missing for the sensor
  std::vector<int> time_id(num_sites, -1);

  // FM TEMPORARY!!!!!!!!!!!!!
  time_id = WGD->time_id;

  if (WID->hrrrInput){
    if (WID->hrrrInput->interpolationScheme != 2){
      // loop to adjust the number of sensors have information for the running timestep of the code
      for (auto i = 0u; i < WID->metParams->sensors.size(); i++) {
	if (time_id[i] == -1 || WGD->hrrrInputData->hrrrSensorUTMx[i] < WID->simParams->UTMx ||
	    WGD->hrrrInputData->hrrrSensorUTMx[i] > WID->simParams->UTMx + (WGD->nx-1)*WGD->dx ||
	    WGD->hrrrInputData->hrrrSensorUTMy[i] < WID->simParams->UTMy || WGD->hrrrInputData->hrrrSensorUTMy[i] > WID->simParams->UTMy + (WGD->ny-1)*WGD->dy)     {
	  num_sites -= 1;
	}
      }

      available_sensor_id.clear();
      for (auto i = 0u; i < WID->metParams->sensors.size(); i++) {
	if (time_id[i] != -1 && WGD->hrrrInputData->hrrrSensorUTMx[i] >= WID->simParams->UTMx &&
	    WGD->hrrrInputData->hrrrSensorUTMx[i] <= WID->simParams->UTMx + (WGD->nx-1)*WGD->dx &&
	    WGD->hrrrInputData->hrrrSensorUTMy[i] >= WID->simParams->UTMy && WGD->hrrrInputData->hrrrSensorUTMy[i] <= WID->simParams->UTMy + (WGD->ny-1)*WGD->dy)   {
	  available_sensor_id.push_back(i);
	}
      }
    }else{
      // loop to adjust the number of sensors have information for the running timestep of the code
      for (auto i = 0u; i < WID->metParams->sensors.size(); i++) {
	if (time_id[i] == -1){
	  num_sites -= 1;
	}
      }
      available_sensor_id.clear();
      for (auto i = 0u; i < WID->metParams->sensors.size(); i++) {
	if (time_id[i] != -1){
	  available_sensor_id.push_back(i);
	}
      } 
    }
  }else{
    // loop to adjust the number of sensors have information for the running timestep of the code
    for (auto i = 0u; i < WID->metParams->sensors.size(); i++) {
      if (time_id[i] == -1){
	num_sites -= 1;
      }
    }
    available_sensor_id.clear();
    for (auto i = 0u; i < WID->metParams->sensors.size(); i++) {
      if (time_id[i] != -1){
	available_sensor_id.push_back(i);
      }
    } 
  }
  
  if (available_sensor_id.size() != 0){
    if (WID->hrrrInput){
      WGD->hrrrInputData->readAloftData(time_id[available_sensor_id[0]]);
    }
  }else{
    std::cout << "Warning!!!! There is no available data for sensor at this timestep" << std::endl;
    return;
  }
  
  std::cout << "Number of sensors with available data:  " << available_sensor_id.size() << std::endl;
  
  u_prof.clear();
  u_prof.resize(num_sites * WGD->nz, 0.0);
  v_prof.clear();
  v_prof.resize(num_sites * WGD->nz, 0.0);

  site_id.clear();
  site_id.resize(num_sites, 0);

  abl_height.clear();

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

    if (WID->hrrrInput){
      if (WID->hrrrInput->interpolationScheme != 2){
	if (WGD->hrrrInputData->hrrrSensorUTMx[i] < WID->simParams->UTMx || WGD->hrrrInputData->hrrrSensorUTMx[i] > WID->simParams->UTMx + (WGD->nx-1)*WGD->dx ||
	    WGD->hrrrInputData->hrrrSensorUTMy[i] < WID->simParams->UTMy || WGD->hrrrInputData->hrrrSensorUTMy[i] > WID->simParams->UTMy + (WGD->ny-1)*WGD->dy) {
	  count += 1;
	  continue;
	}
      }
    }
    
    float convergence = 0.0;

    int idx = i - count;// id of the available sensors for the running timestep of the code

    
    TimeSeries *ts = WID->metParams->sensors[i]->TS[time_id[i]];
    //average_one_overL += ts->site_one_overL / num_sites;

    // THIS SOULD NOT BE HERE!!! AND ITS NOT FINISHED
    if (WID->simParams->UTMx != 0 && WID->simParams->UTMy != 0) {
      if (WID->metParams->sensors[idx]->site_coord_flag == 1) {
        WID->metParams->sensors[idx]->site_UTM_x = WID->metParams->sensors[idx]->site_xcoord * acos(WGD->theta)
                                                 + WID->metParams->sensors[idx]->site_ycoord * asin(WGD->theta)
                                                 + WID->simParams->UTMx;
        WID->metParams->sensors[idx]->site_UTM_y = WID->metParams->sensors[idx]->site_xcoord * asin(WGD->theta)
                                                 + WID->metParams->sensors[idx]->site_ycoord * acos(WGD->theta)
                                                 + WID->simParams->UTMy;
        WID->metParams->sensors[idx]->site_UTM_zone = WID->simParams->UTMZone;
        // Calling UTMConverter function to convert UTM coordinate to lat/lon and vice versa
        GIStool::UTMConverter(WID->metParams->sensors[idx]->site_lon,
                              WID->metParams->sensors[idx]->site_lat,
                              WID->metParams->sensors[idx]->site_UTM_x,
                              WID->metParams->sensors[idx]->site_UTM_y,
                              WID->metParams->sensors[idx]->site_UTM_zone,
                              true,
                              1);
      }

      if (WID->metParams->sensors[idx]->site_coord_flag == 2) {
        // Calling UTMConverter function to convert UTM coordinate to lat/lon and vice versa
        GIStool::UTMConverter(WID->metParams->sensors[idx]->site_lon,
                              WID->metParams->sensors[idx]->site_lat,
                              WID->metParams->sensors[idx]->site_UTM_x,
                              WID->metParams->sensors[idx]->site_UTM_y,
                              WID->metParams->sensors[idx]->site_UTM_zone,
                              true,
                              1);
        WID->metParams->sensors[idx]->site_xcoord = WID->metParams->sensors[idx]->site_UTM_x - WID->simParams->UTMx;
        WID->metParams->sensors[idx]->site_ycoord = WID->metParams->sensors[idx]->site_UTM_y - WID->simParams->UTMy;
      }
      GIStool::getConvergence(WID->metParams->sensors[idx]->site_lon,
                              WID->metParams->sensors[idx]->site_lat,
                              WID->metParams->sensors[idx]->site_UTM_zone,
                              convergence);
    }

    site_theta[idx] = (270.0 - ts->site_wind_dir[0]) * M_PI / 180.0;
    site_i[idx] = WID->metParams->sensors[i]->site_xcoord / WGD->dx;
    site_j[idx] = WID->metParams->sensors[i]->site_ycoord / WGD->dy;
    site_id[idx] = site_i[idx] + site_j[idx] * WGD->nx;

    // Find which HRRR sensors are inside the domain
    if (WID->hrrrInput){
      if ( site_i[idx] >= 0 && site_i[idx] < WGD->nx-1 && site_j[idx] >= 0 && site_j[idx] < WGD->ny-1){
	site_id[idx] = site_id[idx];
      }else if (site_i[idx] < 0 && site_j[idx] >= 0 && site_j[idx] < WGD->ny-1 ){
	site_id[idx] = site_j[idx] * WGD->nx;
      }else if (site_i[idx] < 0 && site_j[idx] < 0 ){
	site_id[idx] = 0;
      }else if (site_i[idx] < 0 && site_j[idx] >= WGD->ny-1 ){
	site_id[idx] = (WGD->ny - 2) * WGD->nx;
      }else if (site_i[idx] >= WGD->nx-1 && site_j[idx] >= 0 && site_j[idx] < WGD->ny-1 ){
	site_id[idx] = WGD->nx - 2 + site_j[idx] * WGD->nx;
      }else if (site_i[idx] >= 0 && site_i[idx] < WGD->nx-1 && site_j[idx] < 0 ){
	site_id[idx] = site_i[idx];
      }else if (site_i[idx] >= 0 && site_i[idx] < WGD->nx-1 && site_j[idx] >= WGD->ny-1 ){
	site_id[idx] = site_i[idx] + (WGD->ny - 2) * WGD->nx;
      }else if (site_i[idx] >= WGD->nx-1 && site_j[idx] < 0 ){
	site_id[idx] = WGD->nx - 2;
      }else if (site_i[idx] >= WGD->nx-1 && site_j[idx] >= WGD->ny-1 ){
	site_id[idx] = WGD->nx - 2 + (WGD->ny - 2) * WGD->nx;
      }else{
	site_id[idx] = 0;
      }
    }

    z_terrain = WGD->z_face[WGD->terrain_face_id[site_id[idx]]];

    float seaLevelBaseHeight = WID->simParams->DTE_heightField->adfMinMax[0];

    if (ts->site_one_overL > 0.0) {
      // Stable boundary layer
      if (WID->hrrrInput && WGD->hrrrInputData->hrrrPBLHeight[WGD->hrrrInputData->hrrrSensorID[i]] > 200){ // if HRRR file available
	abl_height.push_back(WGD->hrrrInputData->hrrrPBLHeight[WGD->hrrrInputData->hrrrSensorID[i]]); // read in PBL height
      }else{
	abl_height.push_back(200);
      }
    } else if (ts->site_one_overL < 0.0) {
      // Unstable boundary layer
      if (WID->hrrrInput && WGD->hrrrInputData->hrrrPBLHeight[WGD->hrrrInputData->hrrrSensorID[i]] > 1000){ // if HRRR file available
	abl_height.push_back(WGD->hrrrInputData->hrrrPBLHeight[WGD->hrrrInputData->hrrrSensorID[i]]); // read in PBL height
      }else{
	abl_height.push_back(1000);
      }
    } else {
      // Neutral boundary layer
      if (WID->hrrrInput && WGD->hrrrInputData->hrrrPBLHeight[WGD->hrrrInputData->hrrrSensorID[i]] > 100){ // if HRRR file available
	abl_height.push_back(WGD->hrrrInputData->hrrrPBLHeight[WGD->hrrrInputData->hrrrSensorID[i]]); // read in PBL height
      }else{
	abl_height.push_back(100);
      }
    }

    if (WID->hrrrInput){
      // If the PBL height at the sensor location (above the sea level) is less than the 925 mb pressure level
      if (seaLevelBaseHeight + WGD->terrain[site_id[idx]] + abl_height[idx] < 762 ){
	WGD->hrrrInputData->hrrrSpeedTop[i] = WGD->hrrrInputData->hrrrSpeed925[i];
	WGD->hrrrInputData->hrrrDirTop[i] = WGD->hrrrInputData->hrrrDir925[i];
      }else if (seaLevelBaseHeight + WGD->terrain[site_id[idx]] + abl_height[idx] < 1450 && seaLevelBaseHeight + WGD->terrain[site_id[idx]] + abl_height[idx] >= 762){// If the PBL height at the sensor location (above the sea level) is less than the 850 mb and higher than 925 mb pressure level
	WGD->hrrrInputData->hrrrSpeedTop[i] = WGD->hrrrInputData->hrrrSpeed850[i];
	WGD->hrrrInputData->hrrrDirTop[i] = WGD->hrrrInputData->hrrrDir850[i];
      }else{// If the PBL height at the sensor location (above the sea level) is less than the 700 mb and higher than 850 mb pressure level
	WGD->hrrrInputData->hrrrSpeedTop[i] = WGD->hrrrInputData->hrrrSpeed700[i];
	WGD->hrrrInputData->hrrrDirTop[i] = WGD->hrrrInputData->hrrrDir700[i];
      }
    }

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
      float u_surf, v_surf, u_abl, v_abl;
      float wind_speed, wind_dir;
      float psi_z0, x_temp_z0;
      for (auto k = WGD->terrain_face_id[site_id[idx]]; k < WGD->nz - 1; ++k) {
        if (k == WGD->terrain_face_id[site_id[idx]]) {
          if (ts->site_z_ref[0] * ts->site_one_overL >= 0) {
            psi = 4.7 * ts->site_z_ref[0] * ts->site_one_overL;
	    psi_z0 = 4.7 * ts->site_z0 * ts->site_one_overL;
          } else {
            x_temp = pow((1.0 - 15.0 * ts->site_z_ref[0] * ts->site_one_overL), 0.25);
	    x_temp_z0 = pow((1.0 - 15.0 * ts->site_z0 * ts->site_one_overL), 0.25);
            psi = -2.0 * log(0.5 * (1.0 + x_temp)) - log(0.5 * (1.0 + pow(x_temp, 2.0))) + 2.0 * atan(x_temp) - 0.5 * M_PI;
	    psi_z0 = -2.0 * log(0.5 * (1.0 + x_temp_z0)) - log(0.5 * (1.0 + pow(x_temp_z0, 2.0))) + 2.0 * atan(x_temp_z0) - 0.5 * M_PI;
          }
	  u_star = ts->site_U_ref[0] * vk / (log((ts->site_z_ref[0] + ts->site_z0) / ts->site_z0) + psi + psi_z0);
	    /*if (idx == 23 || idx == 316){
	    std::cout << "u_star:   " << u_star << std::endl;
	    std::cout << "psi_z0:   " << psi_z0 << std::endl;
	    std::cout << "psi:   " << psi << std::endl;
	    std::cout << "log((ts->site_z_ref[0] + ts->site_z0) / ts->site_z0) + psi + psi_z0:   " << (log((ts->site_z_ref[0] + ts->site_z0) / ts->site_z0) + psi + psi_z0) << std::endl;
	    }*/
        }

        if ((WGD->z[k] - z_terrain) * ts->site_one_overL > 0 && WID->hrrrInput != 0) {// Stable BL
	  surf_layer_height = asl_percent * abl_height[idx];
	  
	  if (surf_layer_height >= ts->site_z_ref[0]){
	    if (WGD->z[k] - z_terrain < 0){
	      continue;
	    }else if ( (WGD->z[k] - z_terrain) <= surf_layer_height ) {// If height (above ground) is less than or equal to ASL height
	      psi = 4.7 * (WGD->z[k] - z_terrain) * ts->site_one_overL;
	      u_prof[idx * WGD->nz + k] = (cos(site_theta[idx]) * u_star / vk) * (log(((WGD->z[k] - z_terrain) + ts->site_z0) / ts->site_z0) + psi + psi_z0);
	      v_prof[idx * WGD->nz + k] = (sin(site_theta[idx]) * u_star / vk) * (log(((WGD->z[k] - z_terrain) + ts->site_z0) / ts->site_z0) + psi + psi_z0);
	      u_surf = u_prof[idx * WGD->nz + k];
	      v_surf = v_prof[idx * WGD->nz + k];
	    }else if ( (WGD->z[k] - z_terrain) > surf_layer_height && (WGD->z[k] - z_terrain) <= abl_height[idx]) {// If height (above ground) is greater than ASL height or less than ABL height
	      u_prof[idx * WGD->nz + k] = ((WGD->z[k] - z_terrain - surf_layer_height)*(WGD->hrrrInputData->hrrrSpeedTop[i]*cos(WGD->hrrrInputData->hrrrDirTop[i]*M_PI/180.0) - u_surf)/(abl_height[idx] - surf_layer_height)) + u_surf;
	      v_prof[idx * WGD->nz + k] = ((WGD->z[k] - z_terrain - surf_layer_height)*(WGD->hrrrInputData->hrrrSpeedTop[i]*sin(WGD->hrrrInputData->hrrrDirTop[i]*M_PI/180.0) - v_surf)/(abl_height[idx] - surf_layer_height)) + v_surf;
	      u_abl = WGD->hrrrInputData->hrrrSpeedTop[i]*cos(WGD->hrrrInputData->hrrrDirTop[i]*M_PI/180.0);
	      v_abl = WGD->hrrrInputData->hrrrSpeedTop[i]*sin(WGD->hrrrInputData->hrrrDirTop[i]*M_PI/180.0);
	    }else{
	      u_prof[idx * WGD->nz + k] = u_abl;
	      v_prof[idx * WGD->nz + k] = v_abl;
	    }
	  }else{
	    if (WGD->z[k] - z_terrain < 0){
	      continue;
	    }else if ( (WGD->z[k] - z_terrain) <= ts->site_z_ref[0]) {// If height (above ground) is less than or equal to ASL height
	      psi = 4.7 * (WGD->z[k] - z_terrain) * ts->site_one_overL;
	      u_prof[idx * WGD->nz + k] = (cos(site_theta[idx]) * u_star / vk) * (log(((WGD->z[k] - z_terrain) + ts->site_z0) / ts->site_z0) + psi + psi_z0);
	      v_prof[idx * WGD->nz + k] = (sin(site_theta[idx]) * u_star / vk) * (log(((WGD->z[k] - z_terrain) + ts->site_z0) / ts->site_z0) + psi + psi_z0);
	    }else if ( (WGD->z[k] - z_terrain) > ts->site_z_ref[0] && (WGD->z[k] - z_terrain) <= abl_height[idx]) {// If height (above ground) is greater than ASL height or less than ABL height
	      u_prof[idx * WGD->nz + k] = ((WGD->z[k] - z_terrain - ts->site_z_ref[0])*(WGD->hrrrInputData->hrrrSpeedTop[i]*cos(WGD->hrrrInputData->hrrrDirTop[i]*M_PI/180.0) - ts->site_U_ref[0]*cos(site_theta[idx]))/(abl_height[idx] - ts->site_z_ref[0])) + ts->site_U_ref[0]*cos(site_theta[idx]);
	      v_prof[idx * WGD->nz + k] = ((WGD->z[k] - z_terrain - ts->site_z_ref[0])*(WGD->hrrrInputData->hrrrSpeedTop[i]*sin(WGD->hrrrInputData->hrrrDirTop[i]*M_PI/180.0) - ts->site_U_ref[0]*sin(site_theta[idx]))/(abl_height[idx] - ts->site_z_ref[0])) + ts->site_U_ref[0]*sin(site_theta[idx]);
	      u_abl = WGD->hrrrInputData->hrrrSpeedTop[i]*cos(WGD->hrrrInputData->hrrrDirTop[i]*M_PI/180.0);
	      v_abl = WGD->hrrrInputData->hrrrSpeedTop[i]*sin(WGD->hrrrInputData->hrrrDirTop[i]*M_PI/180.0);
	    }else{
	      u_prof[idx * WGD->nz + k] = u_abl;
	      v_prof[idx * WGD->nz + k] = v_abl;
	    }
	  }
	}else if ((WGD->z[k] - z_terrain) * ts->site_one_overL == 0 && WID->hrrrInput != 0) {// Neutral BL
	  psi = 4.7 * (WGD->z[k] - z_terrain) * ts->site_one_overL;
	  u_prof[idx * WGD->nz + k] = (cos(site_theta[idx]) * u_star / vk) * (log(((WGD->z[k] - z_terrain) + ts->site_z0) / ts->site_z0) + psi + psi_z0);
	  v_prof[idx * WGD->nz + k] = (sin(site_theta[idx]) * u_star / vk) * (log(((WGD->z[k] - z_terrain) + ts->site_z0) / ts->site_z0) + psi + psi_z0);
        }else if ((WGD->z[k] - z_terrain) * ts->site_one_overL >= 0 && WID->hrrrInput == 0){
	  psi = 4.7 * (WGD->z[k] - z_terrain) * ts->site_one_overL;
	  u_prof[idx * WGD->nz + k] = (cos(site_theta[idx]) * u_star / vk) * (log(((WGD->z[k] - z_terrain) + ts->site_z0) / ts->site_z0) + psi + psi_z0);
	  v_prof[idx * WGD->nz + k] = (sin(site_theta[idx]) * u_star / vk) * (log(((WGD->z[k] - z_terrain) + ts->site_z0) / ts->site_z0) + psi + psi_z0);
	}else{// Unstable BL
          x_temp = pow((1.0 - 15.0 * (WGD->z[k] - z_terrain) * ts->site_one_overL), 0.25);
          psi = -2.0 * log(0.5 * (1.0 + x_temp)) - log(0.5 * (1.0 + pow(x_temp, 2.0))) + 2.0 * atan(x_temp) - 0.5 * M_PI;
	  u_prof[idx * WGD->nz + k] = (cos(site_theta[idx]) * u_star / vk) * (log(((WGD->z[k] - z_terrain) + ts->site_z0) / ts->site_z0) + psi + psi_z0);
	  v_prof[idx * WGD->nz + k] = (sin(site_theta[idx]) * u_star / vk) * (log(((WGD->z[k] - z_terrain) + ts->site_z0) / ts->site_z0) + psi + psi_z0);
        }
	// Free convection limit
	if (abs(u_star) >= 2*ts->site_U_ref[0]){
	  u_prof[idx * WGD->nz + k] = (cos(site_theta[idx]) * ts->site_U_ref[0]);
	  v_prof[idx * WGD->nz + k] = (sin(site_theta[idx]) * ts->site_U_ref[0]);
	}
	/*if (idx == 23 || idx == 316){
	    std::cout << "k:   " << k << std::endl;
	    std::cout << "ts->site_z0:   " << ts->site_z0 << std::endl;
	    std::cout << "site_theta[idx]:   " << site_theta[idx] << std::endl;
	    std::cout << "ts->site_one_overL:   " << ts->site_one_overL << std::endl;
	    std::cout << "u_star:   " << u_star << std::endl;
	    std::cout << "ts->site_U_ref[0]:   " << ts->site_U_ref[0] << std::endl;
	    std::cout << "u_prof[idx * WGD->nz + k]:   " << u_prof[idx * WGD->nz + k] << std::endl;
	    std::cout << "v_prof[idx * WGD->nz + k]:   " << v_prof[idx * WGD->nz + k] << std::endl;
	    }*/
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

  std::cout << "Sensor profiles are created" << std::endl;

  return;
}


void WindProfilerSensorType::singleSensorInterpolation(WINDSGeneralData *WGD)

{
  for (auto k = 0; k < WGD->nz - 1; ++k) {
    //Set the modified k-index (sensor)
    int k_mod_sens = k + WGD->terrain_face_id[site_id[0]];
    for (auto j = 0; j < WGD->ny; ++j) {
      for (auto i = 0; i < WGD->nx; ++i) {

        int id = i + j * WGD->nx;//Index in horizontal surface
        int k_mod(0);
        //If height added to top of terrain is still inside QES domain
        if (k + WGD->terrain_face_id[id] < WGD->nz) {
          //Set the modified k-index (current location)
          k_mod = k + WGD->terrain_face_id[id];
        } else {
          continue;
        }
        // Lineralized index for cell faced values
        int icell_face = i + j * WGD->nx + k_mod * WGD->nx * WGD->ny;
        // If the height difference between the terrain at the curent cell and sensor location is less than ABL height
        if (abs(WGD->z[WGD->terrain_face_id[id]] - WGD->z[WGD->terrain_face_id[site_id[0]]]) > abl_height[id]) {
          surf_layer_height = asl_percent * abl_height[id];
        } else {
          surf_layer_height = asl_percent * (2 * abl_height[id] - (WGD->z[WGD->terrain_face_id[id]] - WGD->z[WGD->terrain_face_id[site_id[0]]]));
        }
        // If height (above ground) is less than or equal to ASL height
        if (WGD->z[k] <= surf_layer_height) {
          WGD->u0[icell_face] = u_prof[k_mod_sens];
          WGD->v0[icell_face] = v_prof[k_mod_sens];
        }// If sum of z index and the terrain index at the sensor location is outside the domain
        else if (k + WGD->terrain_face_id[site_id[0]] > WGD->nz - 2) {
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
