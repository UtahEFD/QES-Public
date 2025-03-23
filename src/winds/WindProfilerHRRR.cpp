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
 * @file WindProfilerHRRR.cpp
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

#include "winds/WINDSInputData.h"
#include "winds/WINDSGeneralData.h"
#include "WindProfilerHRRR.h"


/*
 * Notes, Behnam B,
 * 
 * 1- Double check the interpolation since it creates zero values at higher heights
 * 2- Double check the near edge velocity setting
 * 3- Improve sensor finding algorithm
 * 4- Comment and see the results
 */

void WindProfilerHRRR::interpolateWindProfile(const WINDSInputData *WID, WINDSGeneralData *WGD)
{
  
  sensorsProfiles(WID, WGD);
  
  int num_sites = available_sensor_id.size();
  std::vector<int> site_i(num_sites, 0);
  std::vector<int> site_j(num_sites, 0);
  std::vector<int> near_available_site (WGD->nx*WGD->ny, 0);
  
  for (auto i = 0u; i < num_sites; i++) {
    site_i[i] = WID->metParams->sensors[available_sensor_id[i]]->site_xcoord / WGD->dx;
    site_j[i] = WID->metParams->sensors[available_sensor_id[i]]->site_ycoord / WGD->dy;
    
  }

  int k_mod;//Modified index in z-direction
  int icell_face, ii;
  float z_terrain;

  if (WID->hrrrInput->interpolationScheme == 1){
    auto start_nearest = std::chrono::high_resolution_clock::now();
    for (auto j = 0; j < WGD->ny; j++) {
      for (auto i = 0; i < WGD->nx; i++) {
	int id = i + j * WGD->nx;//Index in horizontal surface
	for (auto jj = 0; jj < num_sites; jj++) {
	  if (WGD->nearest_site_id[id] == available_sensor_id[jj]){
	    near_available_site[id] = jj;
	    break;
	  }
	}
      }
    }
  
    for (auto k = 0; k < WGD->nz - 1; k++) {
      for (auto j = 0; j < WGD->ny; j++) {
	for (auto i = 0; i < WGD->nx; i++) {
	  int id = i + j * WGD->nx;//Index in horizontal surface
	  z_terrain = WGD->z_face[WGD->terrain_face_id[id]];
	  //If height added to top of terrain is still inside QES domain
	  if (k + WGD->terrain_face_id[id] < WGD->nz) {
	    k_mod = k + WGD->terrain_face_id[id];//Set the modified index
	  } else {
	    continue;
	  }

	  icell_face = i + j * WGD->nx + k_mod * WGD->nx * WGD->ny;
	  ii = near_available_site[id];
	  // If the height difference between the terrain at the curent cell and sensor location is less than ABL height
	  if (abs(WGD->z[WGD->terrain_face_id[id]] - WGD->z[WGD->terrain_face_id[site_id[ii]]]) > abl_height[ii]) {
	    surf_layer_height = asl_percent * abl_height[ii];
	  } else {
	    surf_layer_height = asl_percent * (2 * abl_height[ii] - abs(WGD->z[WGD->terrain_face_id[id]] - WGD->z[WGD->terrain_face_id[site_id[ii]]]));
	  }
	  // If sum of z index and the terrain index at the sensor location is outside the domain
	  if (k + WGD->terrain_face_id[site_id[ii]] > WGD->nz - 2) {
	    WGD->u0[icell_face] = u_prof[ii * WGD->nz + WGD->nz - 2];
	    WGD->v0[icell_face] = v_prof[ii * WGD->nz + WGD->nz - 2];
	    WGD->w0[icell_face] = 0.0;
	  }// If height (above ground) is less than or equal to ASL height
	  else if ((WGD->z[k_mod]-z_terrain) <= surf_layer_height) {
	    WGD->u0[icell_face] = u_prof[ii * WGD->nz + k + WGD->terrain_face_id[site_id[ii]]];
	    WGD->v0[icell_face] = v_prof[ii * WGD->nz + k + WGD->terrain_face_id[site_id[ii]]];
	    WGD->w0[icell_face] = 0.0;
	  }// If height (above ground) is greater than ASL height and modified index is inside the domain
	  else if ((WGD->z[k_mod]-z_terrain) > surf_layer_height
		   && k + WGD->terrain_face_id[site_id[ii]] < WGD->nz
		   && k_mod > k + WGD->terrain_face_id[site_id[ii]]) {
	    WGD->u0[icell_face] = u_prof[ii * WGD->nz + k_mod];
	    WGD->v0[icell_face] = v_prof[ii * WGD->nz + k_mod];
	    WGD->w0[icell_face] = 0.0;
	  } else {
	    WGD->u0[icell_face] = u_prof[ii * WGD->nz + k + WGD->terrain_face_id[site_id[ii]]];
	    WGD->v0[icell_face] = v_prof[ii * WGD->nz + k + WGD->terrain_face_id[site_id[ii]]];
	    WGD->w0[icell_face] = 0.0;
	  }	  
	}
      }
    }
    auto finish_nearest = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = finish_nearest - start_nearest;
    std::cout << "Elapsed time for No Interpolation: " << elapsed.count() << " s\n";
  }
 
  int count;
  int id, idx;
  if (WID->hrrrInput->interpolationScheme == 2){
    auto start_biLinear = std::chrono::high_resolution_clock::now();
    for (auto j = 1; j < WGD->ny-1; j++) {
      for (auto i = 1; i < WGD->nx-1; i++) {
	id = i + j * (WGD->nx-1);//Index in horizontal surface
	idx = i + j * WGD->nx;
	//If height added to top of terrain is still inside QES domain
	for (auto k = 0; k < WGD->nz - 1; k++) {
	  if (k + WGD->terrain_face_id[idx] < WGD->nz) {
	    k_mod = k + WGD->terrain_face_id[idx];//Set the modified index
	  } else {
	    continue;
	  }	
	  icell_face = i + j * WGD->nx + k_mod * WGD->nx * WGD->ny;
	  biLinearInterpolation(WID, WGD, i, j, k, k_mod, id, idx, icell_face, site_i, site_j);
	}
      }
    }
    int i = 1;
    for (auto j = 0; j < WGD->ny-1; j++) {
      for (auto k = 1; k < WGD->nz-1; k++){
	icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
	int icell_new = (i-1) + j * WGD->nx + k * WGD->nx * WGD->ny;
	WGD->u0[icell_new] = WGD->u0[icell_face];
	WGD->v0[icell_new] = WGD->v0[icell_face];
      }
    }

    i = WGD->nx-2;
    for (auto j = 0; j < WGD->ny-1; j++) {
      for (auto k = 1; k < WGD->nz-1; k++){
	icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
	int icell_new = (i+1) + j * WGD->nx + k * WGD->nx * WGD->ny;
	WGD->u0[icell_new] = WGD->u0[icell_face];
	WGD->v0[icell_new] = WGD->v0[icell_face];
      }
    }

    int j = 1;
    for (auto i = 1; i < WGD->nx-1; i++) {
      for (auto k = 1; k < WGD->nz-1; k++){
	icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
	int icell_new = i + (j-1) * WGD->nx + k * WGD->nx * WGD->ny;
	WGD->u0[icell_new] = WGD->u0[icell_face];
	WGD->v0[icell_new] = WGD->v0[icell_face];
      }
    }

    j = WGD->ny-2;
    for (auto i = 1; i < WGD->nx-1; i++) {
      for (auto k = 1; k < WGD->nz-1; k++){
	icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
	int icell_new = i + (j+1) * WGD->nx + k * WGD->nx * WGD->ny;
	WGD->u0[icell_new] = WGD->u0[icell_face];
	WGD->v0[icell_new] = WGD->v0[icell_face];
      }
    }
    auto finish_biLinear = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = finish_biLinear - start_biLinear;
    std::cout << "Elapsed time for Bilinear Interpolation: " << elapsed.count() << " s\n";
  }

  return;
}


void WindProfilerHRRR::biLinearInterpolation(const WINDSInputData *WID, WINDSGeneralData *WGD, int i, int j, int k, int k_mod, int id, int idx, int icell_face, std::vector<int> site_i, std::vector<int> site_j)
{
  if (WGD->closest_site_ids[id].size() != 4){
    std::cout << "Not enough HRRR sites"  << std::endl;
    return;
  }

  float r1, r2;
  int site1_id, site2_id, site3_id, site4_id;
  float x1, x2, x3, x4, y1, y2, y3, y4;
  float xi, yj;
  std::vector<float> u, v;
  u.resize(WGD->closest_site_ids[id].size(), 0.0);
  v.resize(WGD->closest_site_ids[id].size(), 0.0);
  
  site1_id = WGD->closest_site_ids[id][0];
  site2_id = WGD->closest_site_ids[id][1];
  site3_id = WGD->closest_site_ids[id][2];
  site4_id = WGD->closest_site_ids[id][3];


  x1 = WID->metParams->sensors[site1_id]->site_xcoord;
  x2 = WID->metParams->sensors[site2_id]->site_xcoord;
  x3 = WID->metParams->sensors[site3_id]->site_xcoord;
  x4 = WID->metParams->sensors[site4_id]->site_xcoord;
  y1 = WID->metParams->sensors[site1_id]->site_ycoord;
  y2 = WID->metParams->sensors[site2_id]->site_ycoord;
  y3 = WID->metParams->sensors[site3_id]->site_ycoord;
  y4 = WID->metParams->sensors[site4_id]->site_ycoord;
  xi = i * WGD->dx;
  yj = j * WGD->dy;
  r1 = ((xi-x1)*(y2-y1)/(x2-x1)) + y1;
  r2 = ((xi-x4)*(y3-y4)/(x3-x4)) + y4;

  float z_terrain;
  int jj;
  
  for (auto ii = 0; ii < WGD->closest_site_ids[id].size(); ii++){
    jj = WGD->closest_site_ids[id][ii];
    
    if ( site_i[jj] >= 0 && site_i[jj] < WGD->nx-1 && site_j[jj] >= 0 && site_j[jj] < WGD->ny-1){
      site_id[jj] = site_id[jj];
    }else if (site_i[jj] < 0 && site_j[jj] >= 0 && site_j[jj] < WGD->ny-1 ){
      site_id[jj] = site_j[jj] * WGD->nx;
    }else if (site_i[jj] < 0 && site_j[jj] < 0 ){
      site_id[jj] = 0;
    }else if (site_i[jj] < 0 && site_j[jj] >= WGD->ny-1 ){
      site_id[jj] = (WGD->ny - 2) * WGD->nx;
    }else if (site_i[jj] >= WGD->nx-1 && site_j[jj] >= 0 && site_j[jj] < WGD->ny-1 ){
      site_id[jj] = WGD->nx - 2 + site_j[jj] * WGD->nx;
    }else if (site_i[jj] >= 0 && site_i[jj] < WGD->nx-1 && site_j[jj] < 0 ){
      site_id[jj] = site_i[jj];
    }else if (site_i[jj] >= 0 && site_i[jj] < WGD->nx-1 && site_j[jj] >= WGD->ny-1 ){
      site_id[jj] = site_i[jj] + (WGD->ny - 2) * WGD->nx;
    }else if (site_i[jj] >= WGD->nx-1 && site_j[jj] < 0 ){
      site_id[jj] = WGD->nx - 2;
    }else if (site_i[jj] >= WGD->nx-1 && site_j[jj] >= WGD->ny-1 ){
      site_id[jj] = WGD->nx - 2 + (WGD->ny - 2) * WGD->nx;
    }else{
      site_id[jj] = 0;
    }
    
    z_terrain = WGD->z_face[WGD->terrain_face_id[idx]];
    
    // If the height difference between the terrain at the curent cell and sensor location is less than ABL height
    if (abs(WGD->z[WGD->terrain_face_id[idx]] - WGD->z_face[WGD->terrain_face_id[site_id[jj]]]) > abl_height[jj]) {
      surf_layer_height = asl_percent * abl_height[jj];
    } else {
      surf_layer_height = asl_percent * (2 * abl_height[jj] - abs(WGD->z[WGD->terrain_face_id[idx]] - WGD->z_face[WGD->terrain_face_id[site_id[jj]]]));
    }
   
    // If sum of z index and the terrain index at the sensor location is outside the domain
    if (k + WGD->terrain_face_id[site_id[jj]] > WGD->nz - 2) {
      u[ii] = u_prof[jj * WGD->nz + WGD->nz - 2];
      v[ii] = v_prof[jj * WGD->nz + WGD->nz - 2];
    }// If height (above ground) is less than or equal to ASL height
    else if ((WGD->z[k_mod]-z_terrain) <= surf_layer_height) {
      u[ii] = u_prof[jj * WGD->nz + k + WGD->terrain_face_id[site_id[jj]]];
      v[ii] = v_prof[jj * WGD->nz + k + WGD->terrain_face_id[site_id[jj]]];
    }// If height (above ground) is greater than ASL height and modified index is inside the domain
    else if ((WGD->z[k_mod]-z_terrain) > surf_layer_height
	     && k + WGD->terrain_face_id[site_id[jj]] < WGD->nz
	     && k_mod > k + WGD->terrain_face_id[site_id[jj]]) {
      u[ii] = u_prof[jj * WGD->nz + k_mod];
      v[ii] = v_prof[jj * WGD->nz + k_mod];
    } else {
      u[ii] = u_prof[jj * WGD->nz + k + WGD->terrain_face_id[site_id[jj]]];
      v[ii] = v_prof[jj * WGD->nz + k + WGD->terrain_face_id[site_id[jj]]];
    }
  }

  WGD->u0[icell_face] = ( (r2-yj) * ( (u[0]*(x2-xi)/(x2-x1)) + (u[1]*(xi-x1)/(x2-x1))) / (r2-r1) ) +
    ( (yj-r1) * ( (u[3]*(x3-xi)/(x3-x4)) + (u[2]*(xi-x4)/(x3-x4))) / (r2-r1) );
  WGD->v0[icell_face] = ( (r2-yj) * ( (v[0]*(x2-xi)/(x2-x1)) + (v[1]*(xi-x1)/(x2-x1))) / (r2-r1) ) +
    ( (yj-r1) * ( (v[3]*(x3-xi)/(x3-x4)) + (v[2]*(xi-x4)/(x3-x4))) / (r2-r1) );
  
  return;
}

