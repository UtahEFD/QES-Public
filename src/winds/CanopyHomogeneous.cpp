/****************************************************************************
 * Copyright (c) 2025 University of Utah
 * Copyright (c) 2025 University of Minnesota Duluth
 *
 * Copyright (c) 2025 Behnam Bozorgmehr
 * Copyright (c) 2025 Jeremy A. Gibbs
 * Copyright (c) 2025 Fabien Margairaz
 * Copyright (c) 2025 Eric R. Pardyjak
 * Copyright (c) 2025 Zachary Patterson
 * Copyright (c) 2025 Rob Stoll
 * Copyright (c) 2025 Lucas Ulmer
 * Copyright (c) 2025 Pete Willemsen
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

/** @file CanopyHomogeneous.cpp */

#include "CanopyHomogeneous.h"

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"

CanopyHomogeneous::CanopyHomogeneous(const std::vector<polyVert> &iSP, float iH, float iBH, float iLAI, int iID)
{
  polygonVertices = iSP;
  H = iH;
  W = 0.0;
  L = 0.0;
  base_height = iBH;
  attenuationCoeff = 2 * iLAI;
  ID = iID;

  height_eff = base_height + H;
}

// set et attenuation coefficient
void CanopyHomogeneous::setCellFlags(const WINDSInputData *WID, WINDSGeneralData *WGD, int tree_id)
{
  // When THIS canopy calls this function, we need to do the
  // following:
  // readCanopy(nx, ny, nz, landuse_flag, num_canopies, lu_canopy_flag,
  // canopy_atten, canopy_top);
  auto [nx, ny, nz] = WGD->domain.getDomainCellNum();
  auto [dx, dy, dz] = WGD->domain.getDomainSize();

  // this function need to be called to defined the boundary of the canopy and the icellflags
  float ray_intersect;
  unsigned int num_crossing, vert_id, start_poly;

  // Find out which cells are going to be inside the polygone
  // Based on Wm. Randolph Franklin, "PNPOLY - Point Inclusion in Polygon Test"
  // Check the center of each cell, if it's inside, set that cell to building
  for (auto j = j_start; j < j_end; j++) {
    // Center of cell y coordinate
    float y_cent = (j + 0.5) * dy;
    for (auto i = i_start; i < i_end; i++) {
      float x_cent = (i + 0.5) * dx;
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
        int icell_2d = WGD->domain.cell2d(i, j);

        if (WGD->icellflag_footprint[icell_2d] == 0) {
          // a  building exist here -> skip
        } else {
          // save the (x,y) location of the canopy
          canopy_cell2D.push_back(icell_2d);
          // set the footprint array for canopy
          WGD->icellflag_footprint[icell_2d] = getCellFlagCanopy();

          // Define start index of the canopy in z-direction
          for (size_t k = 1u; k < WGD->domain.z.size(); k++) {
            if (WGD->terrain[icell_2d] + base_height <= WGD->domain.z[k]) {
              WGD->canopy->canopy_bot_index[icell_2d] = k;
              WGD->canopy->canopy_bot[icell_2d] = WGD->terrain[icell_2d] + base_height;
              WGD->canopy->canopy_base[icell_2d] = WGD->domain.z_face[k];
              break;
            }
          }

          // Define end index of the canopy in z-direction
          for (size_t k = 0u; k < WGD->domain.z.size(); k++) {
            if (WGD->terrain[icell_2d] + H < WGD->domain.z[k + 1]) {
              WGD->canopy->canopy_top_index[icell_2d] = k + 1;
              WGD->canopy->canopy_top[icell_2d] = WGD->terrain[icell_2d] + H;
              break;
            }
          }

          // Define the height of the canopy
          WGD->canopy->canopy_height[icell_2d] = WGD->canopy->canopy_top[icell_2d] - WGD->canopy->canopy_bot[icell_2d];

          // define icellflag @ (x,y) for all z(k) in [k_start...k_end]
          for (auto k = WGD->canopy->canopy_bot_index[icell_2d]; k < WGD->canopy->canopy_top_index[icell_2d]; k++) {
            long icell_3d = WGD->domain.cell(i, j, k);
            if (WGD->icellflag[icell_3d] != 0 && WGD->icellflag[icell_3d] != 2) {
              // Canopy cell
              WGD->icellflag[icell_3d] = getCellFlagCanopy();
              WGD->canopy->canopy_atten_coeff[icell_3d] = attenuationCoeff;
              WGD->canopy->icanopy_flag[icell_3d] = tree_id;
              canopy_cell3D.push_back(icell_3d);
            }
          }
        }// end define icellflag!
      }
    }
  }

  // check if the canopy is well defined
  if (canopy_cell2D.size() == 0) {
    k_start = 0;
    k_end = 0;
  } else {
    k_start = nz - 1;
    k_end = 0;
    for (size_t k = 0u; k < canopy_cell2D.size(); k++) {
      if (WGD->canopy->canopy_bot_index[canopy_cell2D[k]] < k_start)
        k_start = WGD->canopy->canopy_bot_index[canopy_cell2D[k]];
      if (WGD->canopy->canopy_top_index[canopy_cell2D[k]] > k_end)
        k_end = WGD->canopy->canopy_top_index[canopy_cell2D[k]];
    }
  }

  // check of illegal definition.
  if (k_start > k_end) {
    std::cerr << "ERROR in tree definition (k_start > k end)" << std::endl;
    exit(EXIT_FAILURE);
  }

  return;
}

void CanopyHomogeneous::canopyVegetation(WINDSGeneralData *WGD, int canopy_id)
{
  auto [nx, ny, nz] = WGD->domain.getDomainCellNum();
  auto [dx, dy, dz] = WGD->domain.getDomainSize();

  // Apply canopy parameterization
  float avg_atten; /**< average attenuation of the canopy */
  float veg_vel_frac; /**< vegetation velocity fraction */
  int num_atten;

  for (auto icell_2d : canopy_cell2D) {

    if (WGD->canopy->canopy_top[icell_2d] > 0) {
      auto [i, j, k] = WGD->domain.getCellIdx(icell_2d);
      int icell_cent_top = WGD->domain.cellAdd(icell_2d, 0, 0, WGD->canopy->canopy_top_index[icell_2d] - 1);

      // NEED REWORK !!!!!

      // Call the bisection method to find the root
      WGD->canopy->canopy_d[icell_2d] = canopyBisection(WGD->canopy->canopy_ustar[icell_2d],
                                                        WGD->canopy->canopy_z0[icell_2d],
                                                        WGD->canopy->canopy_height[icell_2d],
                                                        WGD->canopy->canopy_atten_coeff[icell_cent_top],
                                                        WGD->vk,
                                                        0.0);
      // std::cout << "WGD->vk:" << WGD->vk << "\n";
      // if (WGD->canopy->canopy_d[icell_2d] >= 10000 || isnan(WGD->canopy->canopy_d[icell_2d])) {
      if (WGD->canopy->canopy_d[icell_2d] > WGD->canopy->canopy_height[icell_2d]) {
        std::cerr << "bisection failed to converge \n"
                  << "\t canopy ID = " << canopy_id << "; Height = " << H << "\n"
                  << "\t WGD->canopy_atten[] = " << WGD->canopy->canopy_atten_coeff[icell_cent_top] << "\n"
                  << "\t WGD->canopy_d[] = " << WGD->canopy->canopy_d[icell_2d] << "\n"
                  << "\t WGD->canopy->canopy_ustar[] = " << WGD->canopy->canopy_ustar[icell_2d] << "\n"
                  << "\t WGD->canopy->canopy_z0[] = " << WGD->canopy->canopy_z0[icell_2d] << std::endl;

        WGD->canopy->canopy_d[icell_2d] = canopySlopeMatch(WGD->canopy->canopy_z0[icell_2d],
                                                           WGD->canopy->canopy_height[icell_2d],
                                                           WGD->canopy->canopy_atten_coeff[icell_cent_top]);
      }

      /**< velocity at the height of the canopy */
      // Local variable - not being used by anything... so
      // commented out for now.
      //
      // float u_H = (WGD->canopy_ustar[id]/WGD->vk)*
      //  log((WGD->canopy_top[id]-WGD->canopy_d[id])/WGD->canopy_z0[id]);

      for (auto k = 1; k < nz - 1; k++) {
        // linear index of current face
        long icell_face = WGD->domain.face(i, j, k);
        // linear index of the current cell
        long icell_cent = WGD->domain.cellAdd(icell_2d, 0, 0, k);
        // relative to the terrain
        float z_rel = WGD->domain.z[k] - WGD->domain.z[WGD->terrain_id[icell_2d]];

        if (z_rel < 0) {
          // below the terrain -> skip
        } else if (z_rel < WGD->canopy->canopy_height[icell_2d]) {

          avg_atten = WGD->canopy->canopy_atten_coeff[icell_cent];

          // FM -> calculate averaged attenuation coef. (TO BE TESTE)
          // check if attenuation below or above is different
          if (WGD->canopy->canopy_atten_coeff[WGD->domain.cellAdd(icell_cent, 0, 0, 1)] != WGD->canopy->canopy_atten_coeff[icell_cent]
              || WGD->canopy->canopy_atten_coeff[WGD->domain.cellAdd(icell_cent, 0, 0, -1)] != WGD->canopy->canopy_atten_coeff[icell_cent]) {
            num_atten = 1;
            if (WGD->canopy->canopy_atten_coeff[WGD->domain.cellAdd(icell_cent, 0, 0, 1)] > 0) {
              avg_atten += WGD->canopy->canopy_atten_coeff[WGD->domain.cellAdd(icell_cent, 0, 0, 1)];
              num_atten += 1;
            }
            if (WGD->canopy->canopy_atten_coeff[WGD->domain.cellAdd(icell_cent, 0, 0, -1)] > 0) {
              avg_atten += WGD->canopy->canopy_atten_coeff[WGD->domain.cellAdd(icell_cent, 0, 0, -1)];
              num_atten += 1;
            }
            avg_atten /= num_atten;
          }

          // correction on the velocity within the canopy
          veg_vel_frac = log((WGD->canopy->canopy_height[icell_2d] - WGD->canopy->canopy_d[icell_2d]) / WGD->canopy->canopy_z0[icell_2d])
                         * exp(avg_atten * ((z_rel / WGD->canopy->canopy_height[icell_2d]) - 1)) / log(z_rel / WGD->canopy->canopy_z0[icell_2d]);
          // check if correction is bound and well defined
          if (veg_vel_frac > 1 || veg_vel_frac < 0 || isnan(veg_vel_frac)) {
            // check if correction is valide (0,1)
            veg_vel_frac = 1;
          }

          // apply parametrization
          WGD->u0[icell_face] *= veg_vel_frac;
          WGD->v0[icell_face] *= veg_vel_frac;

          // at the edge of the canopy need to adjust velocity at the next face
          // use canopy_top to detect the edge (worke with level changes)
          if (i < nx - 2) {
            if (WGD->canopy->canopy_top[WGD->domain.cellAdd(icell_2d, 1, 0, 0)] == 0.0) {
              WGD->u0[WGD->domain.faceAdd(icell_face, 1, 0, 0)] *= veg_vel_frac;
            }
          }
          if (j < ny - 2) {
            if (WGD->canopy->canopy_top[WGD->domain.cellAdd(icell_2d, 0, 1, 0)] == 0.0) {
              WGD->v0[WGD->domain.faceAdd(icell_face, 0, 1, 0)] *= veg_vel_frac;
            }
          }
        } else {
          // correction on the velocity above the canopy
          veg_vel_frac = log((z_rel - WGD->canopy->canopy_d[icell_2d]) / WGD->canopy->canopy_z0[icell_2d])
                         / log(z_rel / WGD->canopy->canopy_z0[icell_2d]);


          // check if correction is bound and well defined
          if (veg_vel_frac > 1 || veg_vel_frac < 0 || isnan(veg_vel_frac)) {
            veg_vel_frac = 1;
          }

          // apply parametrization
          WGD->u0[icell_face] *= veg_vel_frac;
          WGD->v0[icell_face] *= veg_vel_frac;

          // at the edge of the canopy need to adjust velocity at the next face
          // use canopy_top to detect the edge (worke with level changes)
          if (i < nx - 2) {
            if (WGD->canopy->canopy_top[WGD->domain.cellAdd(icell_2d, 1, 0, 0)] == 0.0) {
              WGD->u0[WGD->domain.faceAdd(icell_face, 1, 0, 0)] *= veg_vel_frac;
            }
          }
          if (j < ny - 2) {
            if (WGD->canopy->canopy_top[WGD->domain.cellAdd(icell_2d, 0, 1, 0)] == 0.0) {
              WGD->v0[WGD->domain.cellAdd(icell_face, 0, 1, 0)] *= veg_vel_frac;
            }
          }
        }
      }// end of for(auto k=1; k < WGD->nz-1; k++)
    }
  }
}
