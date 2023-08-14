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

/** @file CanopyIsolatedTree.cpp */

#include "CanopyIsolatedTree.h"

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"

CanopyIsolatedTree::CanopyIsolatedTree(const std::vector<polyVert> &iSP, float iH, float iW, float iBH, float iLAI, int iID)
{
  polygonVertices = iSP;
  H = iH;
  W = iW;
  L = W;
  base_height = iBH;
  LAI = iLAI;
  ID = iID;

  height_eff = base_height + H;
  zMaxLAI = 0.5 * H;
}

// set et attenuation coefficient
void CanopyIsolatedTree::setCellFlags(const WINDSInputData *WID, WINDSGeneralData *WGD, int tree_id)
{
  // When THIS canopy calls this function, we need to do the
  // following:
  // readCanopy(nx, ny, nz, landuse_flag, num_canopies, lu_canopy_flag,
  // canopy_atten, canopy_top);

  // this function need to be called to defined the boundary of the canopy and the icellflags
  float ray_intersect;
  unsigned int num_crossing, vert_id, start_poly;


  // Find out which cells are going to be inside the polygone
  // Based on Wm. Randolph Franklin, "PNPOLY - Point Inclusion in Polygon Test"
  // Check the center of each cell, if it's inside, set that cell to building
  for (auto j = j_start; j < j_end; j++) {
    // Center of cell y coordinate
    float y_cent = (j + 0.5) * WGD->dy;
    for (auto i = i_start; i < i_end; i++) {
      float x_cent = (i + 0.5) * WGD->dx;
      // Node index
      vert_id = 0;
      start_poly = vert_id;
      num_crossing = 0;
      while (vert_id < polygonVertices.size() - 1) {
        if ((polygonVertices[vert_id].y_poly <= y_cent && polygonVertices[vert_id + 1].y_poly > y_cent)
            || (polygonVertices[vert_id].y_poly > y_cent && polygonVertices[vert_id + 1].y_poly <= y_cent)) {
          ray_intersect = (y_cent - polygonVertices[vert_id].y_poly) / (polygonVertices[vert_id + 1].y_poly - polygonVertices[vert_id].y_poly);
          if (x_cent < (polygonVertices[vert_id].x_poly + ray_intersect * (polygonVertices[vert_id + 1].x_poly - polygonVertices[vert_id].x_poly))) {
            num_crossing += 1;
          }
        }
        vert_id += 1;
        if (polygonVertices[vert_id].x_poly == polygonVertices[start_poly].x_poly
            && polygonVertices[vert_id].y_poly == polygonVertices[start_poly].y_poly) {
          vert_id += 1;
          start_poly = vert_id;
        }
      }

      // if num_crossing is odd = cell is oustside of the polygon
      // if num_crossing is even = cell is inside of the polygon
      if ((num_crossing % 2) != 0) {
        int icell_2d = i + j * (WGD->nx - 1);

        if (WGD->icellflag_footprint[icell_2d] == 0) {
          // a  building exist here -> skip
        } else {
          // Define start index of the canopy in z-direction
          for (size_t k = 1u; k < WGD->z.size(); k++) {
            if (WGD->terrain[icell_2d] + base_height <= WGD->z[k]) {
              WGD->canopy->canopy_bot_index[icell_2d] = k;
              WGD->canopy->canopy_bot[icell_2d] = WGD->terrain[icell_2d] + base_height;
              WGD->canopy->canopy_base[icell_2d] = WGD->z_face[k];
              break;
            }
          }

          // Define end index of the canopy in z-direction
          for (size_t k = 0u; k < WGD->z.size(); k++) {
            if (WGD->terrain[icell_2d] + H < WGD->z[k + 1]) {
              WGD->canopy->canopy_top_index[icell_2d] = k + 1;
              WGD->canopy->canopy_top[icell_2d] = WGD->terrain[icell_2d] + H;
              break;
            }
          }

          WGD->icellflag_footprint[icell_2d] = getCellFlagCanopy();

          canopy_cell2D.push_back(icell_2d);
          WGD->canopy->canopy_height[icell_2d] = WGD->canopy->canopy_top[icell_2d] - WGD->canopy->canopy_bot[icell_2d];

          // define icellflag @ (x,y) for all z(k) in [k_start...k_end]
          for (auto k = WGD->canopy->canopy_bot_index[icell_2d]; k < WGD->canopy->canopy_top_index[icell_2d]; k++) {
            int icell_3d = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
            if (WGD->icellflag[icell_3d] != 0 && WGD->icellflag[icell_3d] != 2) {
              // Canopy cell
              WGD->icellflag[icell_3d] = getCellFlagCanopy();
              WGD->canopy->canopy_atten_coeff[icell_3d] = 0.5 * LAI;
              WGD->canopy->icanopy_flag[icell_3d] = tree_id;
              canopy_cell3D.push_back(icell_3d);
            }
          }
        }// end define icellflag!
      }
    }
  }

  if (canopy_cell2D.size() == 0) {
    k_start = 0;
    k_end = 0;
  } else {
    k_start = WGD->nz - 1;
    k_end = 0;
    for (size_t k = 0u; k < canopy_cell2D.size(); k++) {
      if (WGD->canopy->canopy_bot_index[canopy_cell2D[k]] < k_start)
        k_start = WGD->canopy->canopy_bot_index[canopy_cell2D[k]];
      if (WGD->canopy->canopy_top_index[canopy_cell2D[k]] > k_end)
        k_end = WGD->canopy->canopy_top_index[canopy_cell2D[k]];
    }
  }

  if (k_start > k_end) {
    std::cerr << "ERROR in tree definition (k_start > k end)" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (ceil(1.5 * k_end) > WGD->nz - 1) {
    std::cerr << "ERROR domain too short for tree method" << std::endl;
    exit(EXIT_FAILURE);
  }

  return;
}


void CanopyIsolatedTree::canopyVegetation(WINDSGeneralData *WGD, int tree_id)
{

  // apply canopy parameterization
  float avg_atten; /**< average attenuation of the canopy */
  float veg_vel_frac; /**< vegetation velocity fraction */
  //int num_atten;

  for (size_t n = 0u; n < canopy_cell2D.size(); ++n) {
    int icell_2d = canopy_cell2D[n];

    if (WGD->canopy->canopy_top[icell_2d] > 0) {
      int j = (int)((icell_2d) / (WGD->nx - 1));
      int i = icell_2d - j * (WGD->nx - 1);
      int icell_3d = icell_2d + (WGD->canopy->canopy_top_index[icell_2d] - 1) * (WGD->nx - 1) * (WGD->ny - 1);

      // Call the bisection method to find the root
      WGD->canopy->canopy_d[icell_2d] = canopyBisection(WGD->canopy->canopy_ustar[icell_2d],
                                                        WGD->canopy->canopy_z0[icell_2d],
                                                        WGD->canopy->canopy_height[icell_2d],
                                                        WGD->canopy->canopy_atten_coeff[icell_3d],
                                                        WGD->vk,
                                                        0.0);
      // std::cout << "WGD->vk:" << WGD->vk << "\n";
      // std::cout << "WGD->canopy_atten[icell_cent]:" << WGD->canopy_atten[icell_cent] << "\n";
      if (WGD->canopy->canopy_d[icell_2d] == 10000) {
        std::cout << "bisection failed to converge"
                  << "\n";
        std::cout << "TREE1 " << tree_id << " " << H << " " << base_height << std::endl;

        WGD->canopy->canopy_d[icell_2d] = canopySlopeMatch(WGD->canopy->canopy_z0[icell_2d],
                                                           WGD->canopy->canopy_height[icell_2d],
                                                           WGD->canopy->canopy_atten_coeff[icell_3d]);
      }

      // std::cout << building_id << " "  << canopy_ustar[icell_2d] << " "  << canopy_d[icell_2d] << " "  << canopy_z0[icell_2d] << std::endl;

      /**< velocity at the height of the canopy */
      // Local variable - not being used by anything... so
      // commented out for now.
      //
      // float u_H = (WGD->canopy_ustar[id]/WGD->vk)*
      //  log((WGD->canopy_top[id]-WGD->canopy_d[id])/WGD->canopy_z0[id]);

      for (auto k = 1; k < WGD->nz - 1; k++) {
        int icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        float z_rel = WGD->z[k] - WGD->canopy->canopy_base[icell_2d];

        if (WGD->z[k] < WGD->canopy->canopy_base[icell_2d]) {
          // below the terrain or building
        } else if (WGD->z[k] < WGD->canopy->canopy_top[icell_2d]) {
          if (WGD->canopy->canopy_atten_coeff[icell_3d] > 0) {
            icell_3d = icell_2d + k * (WGD->nx - 1) * (WGD->ny - 1);
            avg_atten = WGD->canopy->canopy_atten_coeff[icell_3d];

            // correction on the velocity within the canopy
            veg_vel_frac = log((WGD->canopy->canopy_height[icell_2d] - WGD->canopy->canopy_d[icell_2d]) / WGD->canopy->canopy_z0[icell_2d])
                           * exp(avg_atten * ((z_rel / WGD->canopy->canopy_height[icell_2d]) - 1)) / log(z_rel / WGD->canopy->canopy_z0[icell_2d]);
            // check if correction is bound and well defined
            if (veg_vel_frac > 1 || veg_vel_frac < 0) {
              veg_vel_frac = 1;
            }

            WGD->u0[icell_face] *= veg_vel_frac;
            WGD->v0[icell_face] *= veg_vel_frac;

            // at the edge of the canopy need to adjust velocity at the next face
            // use canopy_top to detect the edge (worke with level changes)
            if (j < WGD->ny - 2) {
              if (WGD->canopy->canopy_top[icell_2d + (WGD->nx - 1)] == 0.0) {
                WGD->v0[icell_face + WGD->nx] *= veg_vel_frac;
              }
            }
            if (i < WGD->nx - 2) {
              if (WGD->canopy->canopy_top[icell_2d + 1] == 0.0) {
                WGD->u0[icell_face + 1] *= veg_vel_frac;
              }
            }
          }
        } else if (WGD->z[k] < 2.0 * WGD->canopy->canopy_top[icell_2d]) {
          // correction on the velocity above the canopy
          float lam = pow((z_rel - WGD->canopy->canopy_height[icell_2d]) / (1.0 * WGD->canopy->canopy_height[icell_2d]), 1.0);
          veg_vel_frac = log((z_rel - WGD->canopy->canopy_d[icell_2d]) / WGD->canopy->canopy_z0[icell_2d])
                         / log(z_rel / WGD->canopy->canopy_z0[icell_2d]);
          veg_vel_frac = (1 - lam) * veg_vel_frac + lam;

          // check if correction is bound and well defined
          if (veg_vel_frac > 1 || veg_vel_frac < 0) {
            veg_vel_frac = 1;
          }

          WGD->u0[icell_face] *= veg_vel_frac;
          WGD->v0[icell_face] *= veg_vel_frac;

          // at the edge of the canopy need to adjust velocity at the next face
          // use canopy_top to detect the edge (worke with level changes)
          if (j < WGD->ny - 2) {
            icell_3d = icell_2d + WGD->canopy->canopy_bot_index[icell_2d] * (WGD->nx - 1) * (WGD->ny - 1);
            if (WGD->canopy->canopy_top[icell_2d + (WGD->nx - 1)] == 0.0) {
              WGD->v0[icell_face + WGD->nx] *= veg_vel_frac;
            }
          }
          if (i < WGD->nx - 2) {
            icell_3d = icell_2d + WGD->canopy->canopy_bot_index[icell_2d] * (WGD->nx - 1) * (WGD->ny - 1);
            if (WGD->canopy->canopy_top[icell_2d + 1] == 0.0) {
              WGD->u0[icell_face + 1] *= veg_vel_frac;
            }
          }
        } else {
          // do nothing far above the tree
        }
      }// end of for(auto k=1; k < WGD->nz-1; k++)
    }
  }

  icell_face = i_building_cent + j_building_cent * WGD->nx + (k_end + 1) * WGD->nx * WGD->ny;
  u0_h = WGD->u0[icell_face];// u velocity at the height of building at the centroid
  v0_h = WGD->v0[icell_face];// v velocity at the height of building at the centroid

  return;
}

void CanopyIsolatedTree::canopyWake(WINDSGeneralData *WGD, int tree_id)
{

  int u_vegwake_flag(0), v_vegwake_flag(0), w_vegwake_flag(0);
  const int wake_stream_coef = 11;
  const int wake_span_coef = 4;
  const float lambda_sq = 0.083;
  const float epsilon = 10e-10;

  float z0;
  float z_b;
  float x_c, y_c, z_c, yw1, yw3, y_norm;
  float x_p, y_p, x_u, y_u, x_v, y_v, x_w, y_w;
  float x_wall, x_wall_u, x_wall_v, x_wall_w, dn_u, dn_v, dn_w;
  float u_defect, u_c, r_center, theta, delta, B_h;
  float ustar_wake(0), ustar_us(0), mag_us(0);

  int kk(0), k_bottom(1), k_top(WGD->nz - 2);
  int icell_cent, icell_face, icell_2d;

  std::vector<float> u0_modified, v0_modified;
  std::vector<int> u0_mod_id, v0_mod_id;

  float Lt = 0.5 * W;
  Lr = H;

  if (k_end == 0)
    return;

  /*
  icell_face = i_building_cent + j_building_cent * WGD->nx + (k_end + 1) * WGD->nx * WGD->ny;
  u0_h = WGD->u0[icell_face];// u velocity at the height of building at the centroid
  v0_h = WGD->v0[icell_face];// v velocity at the height of building at the centroid
  */

  upwind_dir = atan2(v0_h, u0_h);
  mag_us = sqrt(u0_h * u0_h + v0_h * v0_h);

  yw1 = 0.5 * wake_span_coef * H;
  yw3 = -0.5 * wake_span_coef * H;

  y_norm = yw1;

  for (auto k = 1; k <= k_start; k++) {
    k_bottom = k;
    if (base_height <= WGD->z[k])
      break;
  }

  for (auto k = k_start; k < WGD->nz - 2; k++) {
    k_top = k;
    if (1.5 * (H + base_height) < WGD->z[k + 1])
      break;
  }
  k_top++;

  for (auto k = k_start; k < k_end; k++) {
    kk = k;
    if (0.75 * H + base_height <= WGD->z[k])
      break;
  }

  // if the whole tree (defined as center) is in a flow reversal region -> skip the wake
  icell_cent = i_building_cent + j_building_cent * (WGD->nx - 1) + kk * (WGD->nx - 1) * (WGD->ny - 1);
  if (WGD->icellflag[icell_cent] == 3 || WGD->icellflag[icell_cent] == 4 || WGD->icellflag[icell_cent] == 6)
    return;

  // std::cout << "TREE " << building_id << " " << k_end << " " << k_top << " " << kk << std::endl;

  // mathod 1 -> location of upsteam data point (5% of 1/2 building length)
  // method 2 -> displaced log profile
  if (ustar_method == 1) {
    int i = ceil(((-1.05 * Lt) * cos(upwind_dir) - 0.0 * sin(upwind_dir) + building_cent_x) / WGD->dx);
    int j = ceil(((-1.05 * Lt) * sin(upwind_dir) + 0.0 * cos(upwind_dir) + building_cent_y) / WGD->dy);
    int k = k_end - 1;

    // linearized indexes
    icell_2d = i + j * (WGD->nx);
    icell_face = i + j * (WGD->nx) + k * (WGD->nx) * (WGD->ny);

    z0 = WGD->z0_domain_u[icell_2d];

    // upstream velocity
    float utmp = WGD->u0[icell_face];
    float vtmp = WGD->v0[icell_face];
    mag_us = sqrt(utmp * utmp + vtmp * vtmp);
    // height above ground
    z_c = WGD->z[k] - base_height;
    // friction velocity
    ustar_us = mag_us * WGD->vk / (log((z_c + z0) / z0));
    ustar_wake = ustar_us / mag_us;
  } else if (ustar_method == 2) {
    int i = i_building_cent;
    int j = j_building_cent;
    int k = ceil(1.5 * k_end);

    // linearized indexes
    icell_2d = i + j * (WGD->nx - 1);
    icell_face = i + j * (WGD->nx) + k * (WGD->nx) * (WGD->ny);

    // velocity above the canopy
    float utmp = WGD->u0[icell_face];
    float vtmp = WGD->v0[icell_face];
    mag_us = sqrt(utmp * utmp + vtmp * vtmp);
    z_c = WGD->z[k] - base_height;

    ustar_us = mag_us * WGD->vk / (log((z_c + WGD->canopy->canopy_d[icell_2d]) / WGD->canopy->canopy_z0[icell_2d]));
    ustar_wake = ustar_us / mag_us;

    // std::cout << "TREE2 " << building_id << " " << H+base_height << " " << k_top << std::endl;
    // std::cout << z_c << " " << mag_us << " " << ustar_wake << std::endl;
    // std::cout << canopy_d[icell_2d] << " " << canopy_z0[icell_2d] << std::endl;
  } else {
    ustar_wake = 0.323 / 6.15;
  }

  for (auto k = k_top; k >= k_bottom; k--) {

    // absolute z-coord within building above ground
    z_b = WGD->z[k] - base_height;
    // z-coord relative to center of tree (zMaxLAI)
    z_c = z_b - zMaxLAI;

    for (auto y_idx = 1; y_idx < 2 * ceil((yw1 - yw3) / WGD->dxy); ++y_idx) {

      // y-coord relative to center of tree (zMaxLAI)
      y_c = 0.5 * float(y_idx) * WGD->dxy + yw3;

      if (std::abs(y_c) > std::abs(y_norm)) {
        continue;
      } else if (std::abs(y_c) > Lt && std::abs(y_c) <= yw1) {
        // y_cp=y_c-Lt(ibuild)
        // xwall=sqrt((Lt(ibuild)**2.)-(y_cp**2.))
        x_wall = 0;
      } else {
        x_wall = 0;
        // x_wall=sqrt(pow(Lt,2)-pow(y_c,2));
      }

      int x_idx_min = -1;
      for (auto x_idx = 0; x_idx <= 2.0 * ceil(wake_stream_coef * Lr / WGD->dxy); ++x_idx) {
        u_vegwake_flag = 1;
        v_vegwake_flag = 1;
        w_vegwake_flag = 1;

        // x-coord relative to center of tree (zMaxLAI)
        x_c = 0.5 * float(x_idx) * WGD->dxy;

        int i = ceil(((x_c + x_wall) * cos(upwind_dir) - y_c * sin(upwind_dir) + building_cent_x) / WGD->dx) - 1;
        int j = ceil(((x_c + x_wall) * sin(upwind_dir) + y_c * cos(upwind_dir) + building_cent_y) / WGD->dy) - 1;

        //int i = ((x_c + x_wall) * cos(upwind_dir) - y_c * sin(upwind_dir) + building_cent_x) / WGD->dx;
        //int j = ((x_c + x_wall) * sin(upwind_dir) + y_c * cos(upwind_dir) + building_cent_y) / WGD->dy;
        // check if in the domain
        if (i >= WGD->nx - 2 || i <= 0 || j >= WGD->ny - 2 || j <= 0)
          break;

        // linearized indexes
        icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);

        // check if not in canopy/building set start (was x_idx_min < 0) to x_idx_min > 0
        /* old version - > canopy now
	  if (WGD->icellflag[icell_cent] != 0
            && WGD->icellflag[icell_cent] != 2
            && WGD->icellflag[icell_cent] != getCellFlagCanopy()
            && x_idx_min < 0)
          x_idx_min = x_idx;

        if (WGD->icellflag[icell_cent] == 0
            || WGD->icellflag[icell_cent] == 2
            || WGD->icellflag[icell_cent] == getCellFlagCanopy()) {
          // check for canopy/building/terrain that will disrupt the wake
          if (x_idx_min >= 0) {
            if (WGD->canopy->icanopy_flag[icell_cent] == tree_id) {
              x_idx_min = -1;
            } else if (WGD->icellflag[i + j * (WGD->nx - 1) + kk * (WGD->nx - 1) * (WGD->ny - 1)] == 0
                       || WGD->icellflag[i + j * (WGD->nx - 1) + kk * (WGD->nx - 1) * (WGD->ny - 1)] == 2) {
              break;
            } else if (WGD->icellflag[icell_cent] == 0
                       || WGD->icellflag[icell_cent] == 2
                       || WGD->icellflag[icell_cent] == getCellFlagCanopy()) {
              break;
            }
          } else {
            // check the tree is right by a building/terrain -> skip downstream
            if (WGD->icellflag[icell_cent] == 0 || WGD->icellflag[icell_cent] == 2)
              break;
          }
        }
	*/

        if (WGD->icellflag[icell_cent] != 0
            && WGD->icellflag[icell_cent] != 2
            && x_idx_min < 0)
          x_idx_min = x_idx;

        if (WGD->icellflag[icell_cent] == 0
            || WGD->icellflag[icell_cent] == 2
            || WGD->icellflag[icell_cent] == getCellFlagCanopy()) {
          // check for canopy/building/terrain that will disrupt the wake
          if (x_idx_min >= 0) {
            if (WGD->canopy->icanopy_flag[icell_cent] == tree_id) {
              x_idx_min = -1;
            } else if (WGD->icellflag[i + j * (WGD->nx - 1) + kk * (WGD->nx - 1) * (WGD->ny - 1)] == 0
                       || WGD->icellflag[i + j * (WGD->nx - 1) + kk * (WGD->nx - 1) * (WGD->ny - 1)] == 2) {
              break;
            } else if (WGD->icellflag[icell_cent] == 0
                       || WGD->icellflag[icell_cent] == 2) {
              break;
            }
          } else {
            // check the tree is right by a building/terrain -> skip downstream
            if (WGD->icellflag[icell_cent] == 0 || WGD->icellflag[icell_cent] == 2)
              break;
          }
        }

        // NOTE: wake is not applied in the canopy
        if (WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2 && WGD->icellflag[icell_cent] != getCellFlagCanopy()) {

          // START OF WAKE VELOCITY PARAMETRIZATION

          // wake u-values
          // ij coord of u-face
          int i_u = std::round(((x_c + x_wall) * cos(upwind_dir) - y_c * sin(upwind_dir) + building_cent_x) / WGD->dx);
          int j_u = ((x_c + x_wall) * sin(upwind_dir) + y_c * cos(upwind_dir) + building_cent_y) / WGD->dy;
          if (i_u < WGD->nx - 1 && i_u > 0 && j_u < WGD->ny - 1 && j_u > 0) {
            // not rotated relative coordinate of u-face
            x_p = i_u * WGD->dx - building_cent_x;
            y_p = (j_u + 0.5) * WGD->dy - building_cent_y;
            // rotated relative coordinate of u-face
            x_u = x_p * cos(upwind_dir) + y_p * sin(upwind_dir);
            y_u = -x_p * sin(upwind_dir) + y_p * cos(upwind_dir);

            if (std::abs(y_u) > std::abs(y_norm)) {
              break;
            } else {
              x_wall_u = 0;
            }

            // adjusted downstream value
            x_u -= x_wall_u;

            if (std::abs(y_u) < std::abs(y_norm) && std::abs(y_norm) > epsilon && H > epsilon) {
              dn_u = H;
            } else {
              dn_u = 0.0;
            }

            if (x_u > wake_stream_coef * dn_u)
              u_vegwake_flag = 0;

            // linearized indexes
            icell_cent = i_u + j_u * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
            icell_face = i_u + j_u * WGD->nx + k * WGD->nx * WGD->ny;

            if (dn_u > 0.0 && u_vegwake_flag == 1 && WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) {

              // polar coordinate in the wake
              r_center = sqrt(pow(z_c, 2) + pow(y_u, 2));
              theta = atan2(z_c, y_u);

              // FM - ellipse equation:
              B_h = Bfunc(x_u / H);
              delta = (B_h - 1.15) / sqrt(1 - (1 - pow((B_h - 1.15) / (B_h + 1.15), 2)) * pow(cos(theta), 2)) * H;

              // check if within the wake
              if (r_center < 0.5 * delta) {
                // get velocity deficit
                u_c = ucfunc(x_u / H, ustar_wake);
                u_defect = u_c * (exp(-(r_center * r_center) / (lambda_sq * delta * delta)));

                if (u_defect > WGD->canopy->wake_u_defect[icell_face])
                  WGD->canopy->wake_u_defect[icell_face] = u_defect;

                // std::cout << r_center << " " << delta << " " << ustar_wake << " " << u_c << std::endl;
                //  apply parametrization
                //if (std::abs(WGD->u0[icell_face]) >= std::abs(0.2 * cos(upwind_dir) * mag_us)) {
                //  u0_mod_id.push_back(icell_face);
                //  u0_modified.push_back(WGD->u0[icell_face] * (1. - std::abs(u_defect)));
                //}
              }// if (r_center<delta/1)
            }
          }

          // wake v-values
          // ij coord of v-face
          int i_v = ((x_c + x_wall) * cos(upwind_dir) - y_c * sin(upwind_dir) + building_cent_x) / WGD->dx;
          int j_v = std::round(((x_c + x_wall) * sin(upwind_dir) + y_c * cos(upwind_dir) + building_cent_y) / WGD->dy);
          if (i_v < WGD->nx - 1 && i_v > 0 && j_v < WGD->ny - 1 && j_v > 0) {
            // not rotated relative coordinate of v-face
            x_p = (i_v + 0.5) * WGD->dx - building_cent_x;
            y_p = j_v * WGD->dy - building_cent_y;
            // rotated relative coordinate of u-face
            x_v = x_p * cos(upwind_dir) + y_p * sin(upwind_dir);
            y_v = -x_p * sin(upwind_dir) + y_p * cos(upwind_dir);

            if (std::abs(y_v) > std::abs(y_norm)) {
              break;
            } else {
              x_wall_v = 0;
            }

            // adjusted downstream value
            x_v -= x_wall_v;

            if (std::abs(y_v) < std::abs(y_norm) && std::abs(y_norm) > epsilon && H > epsilon) {
              dn_v = H;
            } else {
              dn_v = 0.0;
            }

            if (x_v > wake_stream_coef * dn_v)
              v_vegwake_flag = 0;

            // linearized indexes
            icell_cent = i_v + j_v * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
            icell_face = i_v + j_v * WGD->nx + k * WGD->nx * WGD->ny;

            if (dn_v > 0.0 && v_vegwake_flag == 1 && WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) {

              // polar coordinate in the wake
              r_center = sqrt(pow(z_c, 2) + pow(y_v, 2));
              theta = atan2(z_c, y_v);

              // FM - ellipse equation:
              B_h = Bfunc(x_v / H);
              delta = (B_h - 1.15) / sqrt(1 - (1 - pow((B_h - 1.15) / (B_h + 1.15), 2)) * pow(cos(theta), 2)) * H;

              // check if within the wake
              if (r_center < 0.5 * delta) {
                // get velocity deficit
                u_c = ucfunc(x_v / H, ustar_wake);
                u_defect = u_c * (exp(-(r_center * r_center) / (lambda_sq * delta * delta)));
                // apply parametrization

                if (u_defect > WGD->canopy->wake_v_defect[icell_face])
                  WGD->canopy->wake_v_defect[icell_face] = u_defect;

                //if (std::abs(WGD->v0[icell_face]) >= std::abs(0.2 * sin(upwind_dir) * mag_us)) {
                //  v0_mod_id.push_back(icell_face);
                //  v0_modified.push_back(WGD->v0[icell_face] * (1. - std::abs(u_defect)));
                //}
              }// if (r_center<delta/1)
            }
          }

          // wake celltype w-values
          // ij coord of cell-center
          int i_w = ceil(((x_c + x_wall) * cos(upwind_dir) - y_c * sin(upwind_dir) + building_cent_x) / WGD->dx) - 1;
          int j_w = ceil(((x_c + x_wall) * sin(upwind_dir) + y_c * cos(upwind_dir) + building_cent_y) / WGD->dy) - 1;

          if (i_w < WGD->nx - 1 && i_w > 0 && j_w < WGD->ny - 1 && j_w > 0) {
            // not rotated relative coordinate of cell-center
            x_p = (i_w + 0.5) * WGD->dx - building_cent_x;
            y_p = (j_w + 0.5) * WGD->dy - building_cent_y;
            // rotated relative coordinate of cell-center
            x_w = x_p * cos(upwind_dir) + y_p * sin(upwind_dir);
            y_w = -x_p * sin(upwind_dir) + y_p * cos(upwind_dir);

            if (std::abs(y_w) > std::abs(y_norm)) {
              break;
            } else {
              x_wall_w = 0;
            }

            // adjusted downstream value
            x_w -= x_wall_w;

            if (std::abs(y_w) < std::abs(y_norm) && std::abs(y_norm) > epsilon && H > epsilon) {
              dn_w = H;
            } else {
              dn_w = 0.0;
            }

            if (x_w > wake_stream_coef * dn_w)
              w_vegwake_flag = 0;

            // linearized indexes
            icell_cent = i_w + j_w * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
            // icell_face = i_v + j_v*WGD->nx+k*WGD->nx*WGD->ny;

            if (dn_w > 0.0 && w_vegwake_flag == 1 && WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) {

              // polar coordinate in the wake
              r_center = sqrt(pow(z_c, 2) + pow(y_w, 2));
              theta = atan2(z_c, y_w);

              // FM - ellipse equation:
              B_h = Bfunc(x_w / H);
              delta = (B_h - 1.15) / sqrt(1 - (1 - pow((B_h - 1.15) / (B_h + 1.15), 2)) * pow(cos(theta), 2)) * H;

              // check if within the wake
              if (r_center < 0.5 * delta) {
                // get velocity deficit
                u_c = ucfunc(x_w / H, ustar_wake);
                u_defect = u_c * (exp(-(r_center * r_center) / (lambda_sq * delta * delta)));
                // apply parametrization
                //if (u_defect >= 0.01)
                WGD->icellflag[icell_cent] = getCellFlagWake();
              }// if (r_center<delta/1)
            }
          }
          // if u,v, and w are done -> exit x-loop
          if (u_vegwake_flag == 0 && v_vegwake_flag == 0 && w_vegwake_flag == 0)
            break;
          // END OF WAKE VELOCITY PARAMETRIZATION
        }
      }// end of x-loop (stream-wise)
    }// end of y-loop (span-wise)
  }// end of z-loop

  for (auto x_id = 0u; x_id < u0_mod_id.size(); x_id++) {
    WGD->u0[u0_mod_id[x_id]] = u0_modified[x_id];
  }

  for (auto y_id = 0u; y_id < v0_mod_id.size(); y_id++) {
    WGD->v0[v0_mod_id[y_id]] = v0_modified[y_id];
  }

  u0_mod_id.clear();
  v0_mod_id.clear();
  u0_modified.clear();
  v0_modified.clear();

  return;
}
